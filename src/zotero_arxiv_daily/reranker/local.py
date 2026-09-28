from .base import BaseReranker, register_reranker

import logging
import warnings

import numpy as np


@register_reranker("local")
class LocalReranker(BaseReranker):

    def get_similarity_score(
        self,
        s1: list[str],
        s2: list[str],
    ) -> np.ndarray:

        from sentence_transformers import SentenceTransformer

        # =========================================================
        # 关闭 HuggingFace / SentenceTransformers 的大量日志
        # =========================================================

        if not self.config.executor.debug:

            from transformers.utils import logging as transformers_logging
            from huggingface_hub.utils import logging as hf_logging

            transformers_logging.set_verbosity_error()
            hf_logging.set_verbosity_error()

            logging.getLogger(
                "sentence_transformers"
            ).setLevel(logging.ERROR)

            logging.getLogger(
                "sentence_transformers.SentenceTransformer"
            ).setLevel(logging.ERROR)

            logging.getLogger(
                "transformers"
            ).setLevel(logging.ERROR)

            logging.getLogger(
                "huggingface_hub"
            ).setLevel(logging.ERROR)

            logging.getLogger(
                "huggingface_hub.utils._http"
            ).setLevel(logging.ERROR)

            warnings.filterwarnings(
                "ignore",
                category=FutureWarning,
            )

        # =========================================================
        # 加载模型
        # =========================================================

        encoder = SentenceTransformer(
            self.config.reranker.local.model,
            trust_remote_code=True,
        )

        # =========================================================
        # 编码参数
        # =========================================================

        if self.config.reranker.local.encode_kwargs:

            encode_kwargs = dict(
                self.config.reranker.local.encode_kwargs
            )

        else:

            encode_kwargs = {}

        # 防止一次 batch 太大
        if "batch_size" not in encode_kwargs:

            encode_kwargs["batch_size"] = 32

        # =========================================================
        # 论文侧
        #
        # Jina retrieval:
        # document
        # =========================================================

        document_kwargs = dict(
            encode_kwargs
        )

        document_kwargs["task"] = "retrieval"
        document_kwargs["prompt_name"] = "document"

        paper_features = encoder.encode(
            s1,
            **document_kwargs,
            show_progress_bar=True,
        )

        # =========================================================
        # 查询侧
        #
        # Jina retrieval:
        # query
        # =========================================================

        query_kwargs = dict(
            encode_kwargs
        )

        query_kwargs["task"] = "retrieval"
        query_kwargs["prompt_name"] = "query"

        query_features = encoder.encode(
            s2,
            **query_kwargs,
            show_progress_bar=True,
        )

        # =========================================================
        # 计算相似度
        #
        # 返回：
        #
        #       paper1  paper2  paper3 ...
        # query1
        # query2
        # query3
        #
        # 即 N × N
        # =========================================================

        sim = encoder.similarity(
            paper_features,
            query_features,
        )

        return sim.numpy()