from abc import ABC, abstractmethod
from omegaconf import DictConfig
from ..protocol import Paper, CorpusPaper
import numpy as np
from typing import Type


class BaseReranker(ABC):

    def __init__(self, config: DictConfig):
        self.config = config

    def rerank(
        self,
        candidates: list[Paper],
        corpus: list[CorpusPaper],
    ) -> list[Paper]:
        """
        新的 Reranker 逻辑：

        不再计算：
            IEEE candidates × Zotero corpus

        而是：
            每篇 IEEE paper × 自己所属的 research_direction

        Zotero corpus 参数仍然保留，是为了兼容原项目接口。
        """

        if not candidates:
            return []

        candidate_texts = []
        direction_texts = []

        for paper in candidates:

            title = getattr(
                paper,
                "title",
                ""
            ) or ""

            abstract = getattr(
                paper,
                "abstract",
                ""
            ) or ""

            # 使用标题 + 摘要作为论文语义内容
            candidate_text = (
                f"Title: {title}\n"
                f"Abstract: {abstract}"
            )

            candidate_texts.append(
                candidate_text
            )

            # IEEE Retriever 已经把关键词保存到了
            # research_direction
            direction = getattr(
                paper,
                "research_direction",
                ""
            ) or ""

            direction_texts.append(
                direction
            )

        # 计算：
        # 每篇论文与自己的研究方向之间的相似度
        sim = self.get_similarity_score(
            candidate_texts,
            direction_texts,
        )

        expected_shape = (
            len(candidates),
            len(candidates),
        )

        if sim.shape != expected_shape:
            raise ValueError(
                "Reranker similarity shape mismatch: "
                f"expected {expected_shape}, "
                f"got {sim.shape}"
            )

        # sim 是 N × N
        #
        # 第 i 行：
        #   第 i 篇论文
        #
        # 第 i 列：
        #   第 i 个 research_direction
        #
        # 所以只取对角线。
        scores = np.diag(sim) * 10

        for score, paper in zip(
            scores,
            candidates
        ):

            score = float(score)

            paper.score = score

            # direction_score 专门保存：
            # 论文与自己研究方向的相关度
            paper.direction_score = score

        candidates = sorted(
            candidates,
            key=lambda x: (
                x.score
                if x.score is not None
                else -float("inf")
            ),
            reverse=True,
        )

        return candidates

    @abstractmethod
    def get_similarity_score(
        self,
        s1: list[str],
        s2: list[str],
    ) -> np.ndarray:
        raise NotImplementedError


registered_rerankers = {}


def register_reranker(name: str):

    def decorator(cls):

        registered_rerankers[name] = cls

        return cls

    return decorator


def get_reranker_cls(
    name: str
) -> Type[BaseReranker]:

    if name not in registered_rerankers:

        raise ValueError(
            f"Reranker {name} not found"
        )

    return registered_rerankers[name]