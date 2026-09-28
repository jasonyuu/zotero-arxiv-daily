from loguru import logger
from pyzotero import zotero
from omegaconf import DictConfig, ListConfig

from .utils import glob_match
from .retriever import get_retriever_cls
from .protocol import CorpusPaper

import random
from datetime import datetime

from .reranker import get_reranker_cls
from .construct_email import render_email
from .utils import send_email

from openai import OpenAI
from tqdm import tqdm
from .sci_ranker import EasyScholarRanker


def normalize_path_patterns(
    patterns: list[str] | ListConfig | None,
    config_key: str
) -> list[str] | None:

    if patterns is None:
        return None

    if not isinstance(patterns, (list, ListConfig)):
        raise TypeError(
            f"config.zotero.{config_key} must be a list of glob patterns or null, "
            'for example ["2026/survey/**"]. Single strings are not supported.'
        )

    if any(not isinstance(pattern, str) for pattern in patterns):
        raise TypeError(
            f"config.zotero.{config_key} must contain only glob pattern strings."
        )

    return list(patterns)


class Executor:

    def __init__(self, config: DictConfig):

        self.config = config

        self.include_path_patterns = normalize_path_patterns(
            config.zotero.include_path,
            "include_path"
        )

        self.ignore_path_patterns = normalize_path_patterns(
            config.zotero.ignore_path,
            "ignore_path"
        )

        self.retrievers = {
            source: get_retriever_cls(source)(config)
            for source in config.executor.source
        }

        self.reranker = get_reranker_cls(
            config.executor.reranker
        )(config)

        self.openai_client = OpenAI(
            api_key=config.llm.api.key,
            base_url=config.llm.api.base_url
        )

        # =========================================================
        # EasyScholar SCI 分区
        # =========================================================

        self.sci_ranker = EasyScholarRanker()

    # =============================================================
    # Zotero
    # =============================================================

    def fetch_zotero_corpus(self) -> list[CorpusPaper]:

        logger.info("Fetching zotero corpus")

        zot = zotero.Zotero(
            self.config.zotero.user_id,
            "user",
            self.config.zotero.api_key
        )

        collections = zot.everything(
            zot.collections()
        )

        collections = {
            c["key"]: c
            for c in collections
        }

        corpus = zot.everything(
            zot.items(
                itemType="conferencePaper || journalArticle || preprint"
            )
        )

        corpus = [
            c
            for c in corpus
            if c["data"]["abstractNote"] != ""
        ]

        def get_collection_path(col_key: str) -> str:

            if p := collections[col_key]["data"]["parentCollection"]:
                return (
                    get_collection_path(p)
                    + "/"
                    + collections[col_key]["data"]["name"]
                )

            return collections[col_key]["data"]["name"]

        for c in corpus:

            paths = [
                get_collection_path(col)
                for col in c["data"]["collections"]
            ]

            c["paths"] = paths

        logger.info(
            f"Fetched {len(corpus)} zotero papers"
        )

        return [
            CorpusPaper(
                title=c["data"]["title"],
                abstract=c["data"]["abstractNote"],
                added_date=datetime.strptime(
                    c["data"]["dateAdded"],
                    "%Y-%m-%dT%H:%M:%SZ"
                ),
                paths=c["paths"]
            )
            for c in corpus
        ]

    # =============================================================
    # Zotero filtering
    # =============================================================

    def filter_corpus(
        self,
        corpus: list[CorpusPaper]
    ) -> list[CorpusPaper]:

        if self.include_path_patterns:

            logger.info(
                "Selecting zotero papers matching "
                f"include_path: {self.include_path_patterns}"
            )

            corpus = [
                c
                for c in corpus
                if any(
                    glob_match(path, pattern)
                    for path in c.paths
                    for pattern in self.include_path_patterns
                )
            ]

        if self.ignore_path_patterns:

            logger.info(
                "Excluding zotero papers matching "
                f"ignore_path: {self.ignore_path_patterns}"
            )

            corpus = [
                c
                for c in corpus
                if not any(
                    glob_match(path, pattern)
                    for path in c.paths
                    for pattern in self.ignore_path_patterns
                )
            ]

        if (
            self.include_path_patterns
            or self.ignore_path_patterns
        ):

            samples = random.sample(
                corpus,
                min(5, len(corpus))
            )

            samples = "\n".join(
                [
                    c.title
                    + " - "
                    + "\n".join(c.paths)
                    for c in samples
                ]
            )

            logger.info(
                f"Selected {len(corpus)} zotero papers:\n"
                f"{samples}\n..."
            )

        return corpus

    # =============================================================
    # Paper identity
    # =============================================================

    @staticmethod
    def paper_identity(paper):

        if getattr(paper, "url", None):
            return paper.url.strip().lower()

        return paper.title.strip().lower()

    # =============================================================
    # SCI / relevance sorting
    #
    # SCI:
    # Q1 > Q2 > Q3 > Q4 > 未知
    #
    # 同分区：
    # direction_score
    # >
    # reranker score
    # >
    # publication year
    # =============================================================

    @staticmethod
    def paper_rank_key(paper):

        sci_quartile = getattr(
            paper,
            "sci_quartile",
            None
        )

        if sci_quartile is None:
            sci_score = 0

        else:
            try:
                sci_quartile = int(sci_quartile)

                if sci_quartile == 1:
                    sci_score = 5
                elif sci_quartile == 2:
                    sci_score = 4
                elif sci_quartile == 3:
                    sci_score = 3
                elif sci_quartile == 4:
                    sci_score = 2
                else:
                    sci_score = 0

            except (
                TypeError,
                ValueError
            ):
                sci_score = 0

        direction_score = getattr(
            paper,
            "direction_score",
            None
        )

        if direction_score is None:
            direction_score = 0

        rerank_score = getattr(
            paper,
            "score",
            None
        )

        if rerank_score is None:
            rerank_score = 0

        publication_year = getattr(
            paper,
            "publication_year",
            None
        )

        if publication_year is None:
            publication_year = 0

        return (
            sci_score,
            direction_score,
            rerank_score,
            publication_year
        )

    # =============================================================
    # 每个方向选 1 篇 + 全局再选 N 篇
    # =============================================================

    def select_final_papers(
        self,
        papers,
        keywords
    ):

        n = len(keywords)

        if n == 0:
            return []

        logger.info(
            f"Starting final paper selection: "
            f"{len(papers)} candidates, "
            f"{n} directions"
        )

        selected = []
        selected_ids = set()

        # =========================================================
        # 第一阶段
        #
        # 每个关键词 / 研究方向选择 1 篇
        # =========================================================

        logger.info(
            f"Stage 1: selecting "
            f"1 paper from each of {n} directions"
        )

        for keyword in keywords:

            direction_papers = [
                p
                for p in papers
                if getattr(
                    p,
                    "research_direction",
                    None
                ) == keyword
            ]

            if not direction_papers:

                logger.warning(
                    f"No papers found for direction: "
                    f"{keyword}"
                )

                continue

            direction_papers.sort(
                key=self.paper_rank_key,
                reverse=True
            )

            # =====================================================
            # 找该方向排名最高、且尚未被其他方向选走的论文
            # =====================================================

            selected_paper = None

            for candidate in direction_papers:

                candidate_id = self.paper_identity(
                    candidate
                )

                if candidate_id not in selected_ids:
                    selected_paper = candidate
                    break

            if selected_paper is None:

                logger.warning(
                    f"All papers for direction "
                    f"'{keyword}' were already selected."
                )

                continue

            paper_id = self.paper_identity(
                selected_paper
            )

            selected.append(
                selected_paper
            )

            selected_ids.add(
                paper_id
            )

            logger.info(
                f"[Direction] {keyword} -> "
                f"{selected_paper.title} "
                f"(SCI Q{getattr(selected_paper, 'sci_quartile', '?')})"
            )

        # =========================================================
        # 第二阶段
        #
        # 从所有剩余论文中再选择 N 篇
        # =========================================================

        remaining = [
            p
            for p in papers
            if self.paper_identity(p)
            not in selected_ids
        ]

        remaining.sort(
            key=self.paper_rank_key,
            reverse=True
        )

        additional_count = n

        logger.info(
            f"Stage 2: selecting "
            f"{additional_count} additional global papers"
        )

        for paper in remaining:

            if len(selected) >= 2 * n:
                break

            paper_id = self.paper_identity(
                paper
            )

            if paper_id in selected_ids:
                continue

            selected.append(
                paper
            )

            selected_ids.add(
                paper_id
            )

            logger.info(
                f"[Global] {paper.title} "
                f"(SCI Q{getattr(paper, 'sci_quartile', '?')})"
            )

        # =========================================================
        # 最终结果
        # =========================================================

        logger.info(
            f"Final paper count: "
            f"{len(selected)} / {2 * n}"
        )

        return selected

    # =============================================================
    # Main
    # =============================================================

    def run(self):

        # =========================================================
        # 1. Zotero corpus
        # =========================================================

        corpus = self.fetch_zotero_corpus()

        corpus = self.filter_corpus(
            corpus
        )

        if len(corpus) == 0:

            logger.error(
                "No zotero papers found. "
                "Please check your zotero settings:\n"
                f"{self.config.zotero}"
            )

            return

        # =========================================================
        # 2. Retrieve IEEE papers
        # =========================================================

        all_papers = []

        for source, retriever in self.retrievers.items():

            logger.info(
                f"Retrieving {source} papers..."
            )

            papers = retriever.retrieve_papers()

            if len(papers) == 0:

                logger.info(
                    f"No {source} papers found"
                )

                continue

            logger.info(
                f"Retrieved {len(papers)} "
                f"{source} papers"
            )

            all_papers.extend(
                papers
            )

        logger.info(
            f"Total {len(all_papers)} papers "
            f"retrieved from all sources"
        )

        # =========================================================
        # 3. Reranking
        # =========================================================

        reranked_papers = []

        if len(all_papers) > 0:

            logger.info(
                "Reranking papers..."
            )

            reranked_papers = self.reranker.rerank(
                all_papers,
                corpus
            )

            logger.info(
                f"Reranker returned "
                f"{len(reranked_papers)} papers"
            )

            # =====================================================
            # EasyScholar SCI 分区
            # =====================================================

            logger.info(
                "Querying EasyScholar SCI quartiles..."
            )

            for paper in tqdm(
                reranked_papers,
                desc="EasyScholar SCI"
            ):

                self.sci_ranker.enrich_paper(
                    paper
                )

        elif not self.config.executor.send_empty:

            logger.info(
                "No new papers found. "
                "No email will be sent."
            )

            return

        # =========================================================
        # 4. 最终筛选
        #
        # N = keyword 数量
        #
        # 候选：
        #     N × 10
        #
        # 最终：
        #     N + N = 2N
        # =========================================================

        if reranked_papers:

            if "ieee" not in self.retrievers:

                logger.error(
                    "IEEE retriever is not configured."
                )

                return

            keywords = list(
                self.retrievers["ieee"]
                .retriever_config
                .keywords
            )

            logger.info(
                f"IEEE keyword count: "
                f"{len(keywords)}"
            )

            logger.info(
                f"Expected final paper count: "
                f"{len(keywords) * 2}"
            )

            reranked_papers = (
                self.select_final_papers(
                    reranked_papers,
                    keywords
                )
            )

        # =========================================================
        # 5. 只对最终论文调用 LLM
        # =========================================================

        if reranked_papers:

            logger.info(
                "Generating TLDR and affiliations "
                f"for {len(reranked_papers)} final papers..."
            )

            for p in tqdm(
                reranked_papers,
                desc="Generating LLM content"
            ):

                p.generate_tldr(
                    self.openai_client,
                    self.config.llm
                )

                p.generate_affiliations(
                    self.openai_client,
                    self.config.llm
                )

        # =========================================================
        # 6. Send email
        # =========================================================

        logger.info(
            "Sending email..."
        )

        email_content = render_email(
            reranked_papers
        )

        send_email(
            self.config,
            email_content
        )

        logger.info(
            "Email sent successfully"
        )