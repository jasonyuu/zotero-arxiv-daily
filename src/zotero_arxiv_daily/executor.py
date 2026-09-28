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

        # =========================================================
        # Zotero collection filtering
        # =========================================================

        self.include_path_patterns = normalize_path_patterns(
            config.zotero.include_path,
            "include_path"
        )

        self.ignore_path_patterns = normalize_path_patterns(
            config.zotero.ignore_path,
            "ignore_path"
        )

        # =========================================================
        # Retrievers
        # =========================================================

        self.retrievers = {
            source: get_retriever_cls(source)(config)
            for source in config.executor.source
        }

        # =========================================================
        # Reranker
        # =========================================================

        self.reranker = get_reranker_cls(
            config.executor.reranker
        )(config)

        # =========================================================
        # OpenAI / LLM
        # =========================================================

        self.openai_client = OpenAI(
            api_key=config.llm.api.key,
            base_url=config.llm.api.base_url
        )

        # =========================================================
        # EasyScholar SCI ranking
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

            if corpus:

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

            else:

                logger.info(
                    "Selected 0 zotero papers after filtering."
                )

        return corpus

    # =============================================================
    # Paper identity
    # =============================================================

    @staticmethod
    def paper_identity(paper):

        url = getattr(
            paper,
            "url",
            None
        )

        if url:

            return url.strip().lower()

        title = getattr(
            paper,
            "title",
            ""
        ) or ""

        return title.strip().lower()

    # =============================================================
    # SCI / relevance sorting
    #
    # SCI:
    # Q1 > Q2 > Q3 > Q4 > 未知
    #
    # 同 SCI 分区：
    # direction_score
    # >
    # reranker score
    # >
    # publication year
    # =============================================================

    @staticmethod
    def paper_rank_key(paper):

        # =========================================================
        # SCI quartile
        # =========================================================

        sci_quartile = getattr(
            paper,
            "sci_quartile",
            None
        )

        if sci_quartile is None:

            sci_score = 0

        else:

            try:

                sci_quartile = int(
                    sci_quartile
                )

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

        # =========================================================
        # Direction relevance
        # =========================================================

        direction_score = getattr(
            paper,
            "direction_score",
            None
        )

        if direction_score is None:
            direction_score = 0

        try:
            direction_score = float(
                direction_score
            )
        except (
            TypeError,
            ValueError
        ):
            direction_score = 0

        # =========================================================
        # General reranker score
        # =========================================================

        rerank_score = getattr(
            paper,
            "score",
            None
        )

        if rerank_score is None:
            rerank_score = 0

        try:
            rerank_score = float(
                rerank_score
            )
        except (
            TypeError,
            ValueError
        ):
            rerank_score = 0

        # =========================================================
        # Publication year
        # =========================================================

        publication_year = getattr(
            paper,
            "publication_year",
            None
        )

        if publication_year is None:
            publication_year = 0

        try:
            publication_year = int(
                publication_year
            )
        except (
            TypeError,
            ValueError
        ):
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

        # =========================================================
        # 清理关键词
        # =========================================================

        keywords = [
            str(keyword).strip()
            for keyword in keywords
            if keyword is not None
            and str(keyword).strip()
        ]

        n = len(keywords)

        if n == 0:

            logger.warning(
                "No IEEE keywords configured."
            )

            return []

        expected_candidates = n * 10
        expected_final = n * 2

        logger.info(
            "=================================================="
        )

        logger.info(
            "Final paper selection"
        )

        logger.info(
            f"Keyword count: {n}"
        )

        logger.info(
            f"Expected IEEE candidate count: "
            f"{expected_candidates}"
        )

        logger.info(
            f"Actual candidate count: "
            f"{len(papers)}"
        )

        logger.info(
            f"Expected final paper count: "
            f"{expected_final}"
        )

        logger.info(
            "=================================================="
        )

        # =========================================================
        # 如果 IEEE Retriever 没有返回预期数量，给出警告
        # =========================================================

        if len(papers) != expected_candidates:

            logger.warning(
                "IEEE candidate count differs from expected: "
                f"expected={expected_candidates}, "
                f"actual={len(papers)}"
            )

        selected = []

        selected_ids = set()

        # =========================================================
        # 第一阶段
        #
        # 每个关键词选择 1 篇
        # =========================================================

        logger.info(
            f"Stage 1: selecting 1 paper "
            f"from each of {n} directions"
        )

        for keyword in keywords:

            direction_papers = [
                p
                for p in papers
                if (
                    getattr(
                        p,
                        "research_direction",
                        None
                    )
                    == keyword
                )
            ]

            if not direction_papers:

                logger.warning(
                    f"No papers found for direction: "
                    f"{keyword}"
                )

                continue

            # -----------------------------------------------------
            # 当前方向内部排序
            # -----------------------------------------------------

            direction_papers.sort(
                key=self.paper_rank_key,
                reverse=True
            )

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

            quartile = getattr(
                selected_paper,
                "sci_quartile",
                None
            )

            quartile_text = (
                f"Q{quartile}"
                if quartile is not None
                else "Unknown"
            )

            logger.info(
                f"[Direction] "
                f"{keyword} -> "
                f"{selected_paper.title} | "
                f"SCI={quartile_text} | "
                f"direction_score="
                f"{getattr(selected_paper, 'direction_score', 0):.4f}"
            )

        # =========================================================
        # 第二阶段
        #
        # 从剩余所有论文中选择 N 篇
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

        logger.info(
            f"Stage 2: selecting "
            f"{n} additional global papers"
        )

        for paper in remaining:

            if len(selected) >= expected_final:

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

            quartile = getattr(
                paper,
                "sci_quartile",
                None
            )

            quartile_text = (
                f"Q{quartile}"
                if quartile is not None
                else "Unknown"
            )

            logger.info(
                f"[Global] "
                f"{paper.title} | "
                f"SCI={quartile_text} | "
                f"direction="
                f"{getattr(paper, 'research_direction', '?')} | "
                f"direction_score="
                f"{getattr(paper, 'direction_score', 0):.4f}"
            )

        # =========================================================
        # 最终检查
        # =========================================================

        if len(selected) < expected_final:

            logger.warning(
                f"Could not reach expected final count: "
                f"{len(selected)} / {expected_final}"
            )

        else:

            logger.info(
                f"Final paper count: "
                f"{len(selected)} / {expected_final}"
            )

        logger.info(
            "=================================================="
        )

        return selected

    # =============================================================
    # Main
    # =============================================================

    def run(self):

        # =========================================================
        # 1. Zotero corpus
        #
        # 注意：
        # Zotero 仍然读取。
        #
        # 但新的 reranker 不再拿 Zotero corpus
        # 与 IEEE 候选论文做 60 × 1429 的相似度计算。
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
        # 2. Retrieve papers
        #
        # IEEE:
        #
        # 6 keywords × 10 papers
        # = 60 candidates
        #
        # 不在这里使用 executor.max_paper_num 截断。
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
        #
        # 新逻辑：
        #
        # IEEE paper
        #      ↓
        # 自己的 research_direction
        #
        # 不再：
        #
        # IEEE paper
        #      ↓
        # 1429 Zotero papers
        # =========================================================

        reranked_papers = []

        if len(all_papers) > 0:

            logger.info(
                "Reranking papers by research direction..."
            )

            reranked_papers = self.reranker.rerank(
                all_papers,
                corpus
            )

            logger.info(
                f"Reranker returned "
                f"{len(reranked_papers)} papers"
            )

        elif not self.config.executor.send_empty:

            logger.info(
                "No new papers found. "
                "No email will be sent."
            )

            return

        # =========================================================
        # 4. EasyScholar SCI quartile
        #
        # 对全部候选论文查询。
        #
        # 注意：
        # 这里只是查询和排序。
        # LLM 还没有调用。
        # =========================================================

        if reranked_papers:

            logger.info(
                "Querying EasyScholar SCI quartiles "
                f"for {len(reranked_papers)} candidates..."
            )

            for paper in tqdm(
                reranked_papers,
                desc="EasyScholar SCI"
            ):

                self.sci_ranker.enrich_paper(
                    paper
                )

            logger.info(
                "EasyScholar SCI enrichment completed."
            )

        # =========================================================
        # 5. Final selection
        #
        # N 个关键词
        #
        # 第一阶段：
        #     N 篇
        #
        # 第二阶段：
        #     N 篇
        #
        # 最终：
        #     2N 篇
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
        # 6. 只对最终论文调用 LLM
        #
        # 例如：
        #
        # 6 keywords
        # → 12 papers
        # → 只有这 12 篇进入 LLM
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

        elif not self.config.executor.send_empty:

            logger.info(
                "No final papers selected. "
                "No email will be sent."
            )

            return

        # =========================================================
        # 7. Send email
        # =========================================================

        logger.info(
            f"Preparing email for "
            f"{len(reranked_papers)} papers..."
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