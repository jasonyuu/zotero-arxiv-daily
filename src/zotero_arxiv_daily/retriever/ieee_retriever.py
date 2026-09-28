import os
import time
from datetime import datetime

import requests
from loguru import logger

from .base import BaseRetriever, register_retriever
from ..protocol import Paper


@register_retriever("ieee")
class IEEERetriever(BaseRetriever):

    API_URL = "https://ieeexploreapi.ieee.org/api/v1/search/articles"

    def __init__(self, config):
        super().__init__(config)

        # =================================================
        # IEEE API Key
        # =================================================

        self.api_key = os.getenv("IEEE_API_KEY")

        if not self.api_key:
            raise ValueError(
                "IEEE_API_KEY environment variable is missing"
            )

        # =================================================
        # 每个关键词最多获取多少篇
        # =================================================

        self.max_records = 10

        # =================================================
        # 自动计算近五年
        #
        # 2026 -> 2022~2026
        # 2027 -> 2023~2027
        # 2028 -> 2024~2028
        # =================================================

        current_year = datetime.now().year

        self.start_year = current_year - 4
        self.end_year = current_year

        logger.info(
            f"IEEE search year range: "
            f"{self.start_year}-{self.end_year}"
        )

        # =================================================
        # HTTP 设置
        # =================================================

        self.timeout = 30

        # 两个关键词请求之间稍微等待一下
        self.request_interval = 1.0

    # =====================================================
    # 获取 IEEE 原始论文
    # =====================================================

    def _retrieve_raw_papers(self):

        raw_papers = []

        keywords = self.retriever_config.keywords

        # 用 DOI / publication_number / title 去重
        seen_ids = set()

        for keyword in keywords:

            logger.info(
                f"Searching IEEE: {keyword}"
            )

            params = {
                "apikey": self.api_key,

                "querytext": keyword,

                # 自动计算的近五年
                "start_year": self.start_year,
                "end_year": self.end_year,

                # 每个关键词最多10篇
                "max_records": self.max_records,

                # 从第一条开始
                "start_record": 1,
            }

            data = self._request(params)

            if data is None:
                continue

            articles = data.get(
                "articles",
                []
            )

            logger.info(
                f"IEEE {keyword}: "
                f"{len(articles)} papers"
            )

            # =================================================
            # 去重
            # =================================================

            for article in articles:

                article_id = (
                    article.get("doi")
                    or article.get("publication_number")
                    or article.get("article_number")
                    or article.get("title")
                )

                if not article_id:
                    continue

                article_id = str(
                    article_id
                ).strip().lower()

                if article_id in seen_ids:
                    continue

                seen_ids.add(article_id)

                raw_papers.append(article)

            time.sleep(
                self.request_interval
            )

        # =====================================================
        # 按发表年份从新到旧排序
        # =====================================================

        def get_year(article):

            value = article.get(
                "publication_year",
                0
            )

            try:
                return int(value)

            except (
                TypeError,
                ValueError
            ):
                return 0

        raw_papers.sort(
            key=get_year,
            reverse=True
        )

        logger.info(
            f"IEEE total unique papers: "
            f"{len(raw_papers)}"
        )

        return raw_papers

    # =====================================================
    # IEEE API 请求
    # =====================================================

    def _request(self, params):

        max_retries = 4

        for attempt in range(
            1,
            max_retries + 1
        ):

            try:

                response = requests.get(
                    self.API_URL,
                    params=params,
                    timeout=self.timeout,
                    headers={
                        "Accept": "application/json",
                        "User-Agent": (
                            "zotero-arxiv-daily/1.0 "
                            "(IEEE Xplore Metadata API client)"
                        ),
                    },
                )

                # =================================================
                # HTTP 200
                # =================================================

                if response.status_code == 200:

                    return response.json()

                # =================================================
                # 这些状态码进行重试
                # =================================================

                if response.status_code in (
                    418,
                    429,
                    500,
                    502,
                    503,
                    504,
                ):

                    wait_seconds = 5 * attempt

                    logger.warning(
                        f"IEEE API returned HTTP "
                        f"{response.status_code}. "
                        f"Retry "
                        f"{attempt}/{max_retries} "
                        f"after "
                        f"{wait_seconds}s."
                    )

                    if attempt < max_retries:

                        time.sleep(
                            wait_seconds
                        )

                        continue

                # =================================================
                # 其他错误
                # =================================================

                logger.warning(
                    f"IEEE request failed: "
                    f"HTTP {response.status_code} - "
                    f"{response.text[:500]}"
                )

                return None

            except requests.RequestException as e:

                wait_seconds = 5 * attempt

                logger.warning(
                    f"IEEE request exception: "
                    f"{e}. Retry "
                    f"{attempt}/{max_retries} "
                    f"after {wait_seconds}s."
                )

                if attempt < max_retries:

                    time.sleep(
                        wait_seconds
                    )

                    continue

                return None

            except ValueError as e:

                logger.warning(
                    f"IEEE JSON decode failed: "
                    f"{e}"
                )

                return None

        return None

    # =====================================================
    # IEEE 原始数据转换为 Paper
    # =====================================================

    def convert_to_paper(
        self,
        raw_paper
    ):

        # =================================================
        # 标题
        # =================================================

        title = raw_paper.get(
            "title",
            ""
        )

        if not title:
            return None

        # =================================================
        # 作者
        # =================================================

        authors = []

        author_info = raw_paper.get(
            "authors",
            {}
        )

        for author in author_info.get(
            "authors",
            []
        ):

            name = author.get(
                "full_name"
            )

            if name:
                authors.append(name)

        # =================================================
        # 摘要
        # =================================================

        abstract = raw_paper.get(
            "abstract",
            ""
        )

        # =================================================
        # IEEE Xplore 页面
        # =================================================

        url = raw_paper.get(
            "html_url",
            ""
        )

        if not url:

            url = raw_paper.get(
                "abstract_url",
                ""
            )

        # =================================================
        # PDF
        # =================================================

        pdf_url = raw_paper.get(
            "pdf_url"
        )

        # =================================================
        # 转换成项目统一 Paper
        # =================================================

        return Paper(
            source=self.name,
            title=title,
            authors=authors,
            abstract=abstract,
            url=url,
            pdf_url=pdf_url,
            full_text=None
        )
