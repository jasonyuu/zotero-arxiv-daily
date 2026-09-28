import os
import time
from datetime import datetime

import requests
from loguru import logger

from .base import BaseRetriever, register_retriever
from ..protocol import Paper


@register_retriever("ieee")
class IEEERetriever(BaseRetriever):

    API_URL = (
        "https://ieeexploreapi.ieee.org/"
        "api/v1/search/articles"
    )

    def __init__(self, config):
        super().__init__(config)

        # =========================================================
        # IEEE API Key
        # =========================================================

        self.api_key = os.getenv("IEEE_API_KEY")

        if not self.api_key:
            raise ValueError(
                "IEEE_API_KEY environment variable is missing"
            )

        # =========================================================
        # 每个关键词最多获取多少篇
        # =========================================================

        self.max_records = 10

        # =========================================================
        # 动态滚动近五年
        #
        # 2026 -> 2022 ~ 2026
        # 2027 -> 2023 ~ 2027
        # 2028 -> 2024 ~ 2028
        # =========================================================

        current_year = datetime.now().year

        self.start_year = current_year - 4
        self.end_year = current_year

        logger.info(
            f"IEEE search year range: "
            f"{self.start_year}-{self.end_year}"
        )

        # =========================================================
        # HTTP 设置
        # =========================================================

        self.timeout = 30

        # IEEE API 请求之间的基础间隔
        #
        # 不要设置得太短。
        # 这里使用 2 秒，避免连续请求触发 QPS 限制。
        #
        # 6 个关键词正常情况下大约需要 12 秒以上。
        # =========================================================

        self.request_interval = 2.0

        # =========================================================
        # IEEE 限流后的重试设置
        # =========================================================

        self.max_retries = 3

        # 遇到 418 / 429 / Service Over Qps 时：
        #
        # 第一次：等待 10 秒
        # 第二次：等待 20 秒
        # 第三次：等待 40 秒
        #
        # 不并发，不绕过 IEEE 限流。
        # =========================================================

        self.retry_backoff = [10, 20, 40]

        # 上一次请求时间
        self._last_request_time = 0.0

    # =============================================================
    # 请求前限速
    # =============================================================

    def _wait_before_request(self):
        """
        确保两个 IEEE API 请求之间至少间隔 request_interval 秒。
        """

        elapsed = time.monotonic() - self._last_request_time

        if elapsed < self.request_interval:
            wait_time = self.request_interval - elapsed

            logger.info(
                f"Waiting {wait_time:.1f}s before next IEEE request..."
            )

            time.sleep(wait_time)

    # =============================================================
    # 获取 IEEE 原始论文
    # =============================================================

    def _retrieve_raw_papers(self):

        raw_papers = []

        keywords = list(self.retriever_config.keywords)

        if not keywords:
            logger.warning(
                "No IEEE keywords configured."
            )
            return raw_papers

        # =========================================================
        # 重要：
        #
        # 这里故意不做全局去重。
        #
        # 用户要求：
        #
        # N 个关键词
        # 每个关键词 10 篇
        # = N * 10 个候选
        #
        # 同一篇论文可能同时出现在多个关键词中。
        # 这种重复不能在这里删除，否则无法保证候选池数量。
        #
        # 最终选择阶段再进行论文级去重。
        # =========================================================

        for index, keyword in enumerate(keywords, start=1):

            logger.info(
                f"[{index}/{len(keywords)}] "
                f"Searching IEEE: {keyword}"
            )

            params = {
                "apikey": self.api_key,
                "querytext": keyword,

                # 动态近五年
                "start_year": self.start_year,
                "end_year": self.end_year,

                # 每个关键词最多 10 篇
                "max_records": self.max_records,

                # 从第一条开始
                "start_record": 1,
            }

            data = self._request(params)

            if data is None:
                logger.warning(
                    f"IEEE keyword '{keyword}' failed."
                )
                continue

            articles = data.get(
                "articles",
                []
            )

            total_records = data.get(
                "total_records",
                0
            )

            logger.info(
                f"IEEE keyword '{keyword}': "
                f"{len(articles)} papers returned, "
                f"{total_records} total matches"
            )

            # =====================================================
            # 给每篇论文标记研究方向
            # =====================================================

            keyword_count = 0

            for article in articles:

                if not isinstance(article, dict):
                    continue

                title = article.get(
                    "title",
                    ""
                )

                if not title:
                    continue

                article["_research_direction"] = keyword

                raw_papers.append(article)

                keyword_count += 1

            logger.info(
                f"IEEE keyword '{keyword}': "
                f"{keyword_count} candidates added"
            )

        # =========================================================
        # 按出版年份倒序
        # =========================================================

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

        # =========================================================
        # 统计
        # =========================================================

        expected_count = (
            len(keywords) * self.max_records
        )

        logger.info(
            f"IEEE candidate pool: "
            f"{len(raw_papers)} papers"
        )

        logger.info(
            f"IEEE expected candidate pool: "
            f"{expected_count} papers"
        )

        if len(raw_papers) < expected_count:

            logger.warning(
                f"IEEE returned fewer candidates than expected: "
                f"{len(raw_papers)}/{expected_count}"
            )

        return raw_papers

    # =============================================================
    # IEEE API 请求
    # =============================================================

    def _request(self, params):

        for attempt in range(self.max_retries + 1):

            # =====================================================
            # 请求前限速
            # =====================================================

            self._wait_before_request()

            try:

                logger.info(
                    f"IEEE API request "
                    f"(attempt {attempt + 1}/{self.max_retries + 1})"
                )

                response = requests.get(
                    self.API_URL,
                    params=params,
                    timeout=self.timeout,
                    headers={
                        "Accept": "application/json",
                        "User-Agent": "Mozilla/5.0",
                    },
                )

                # 记录请求完成时间
                self._last_request_time = time.monotonic()

                # =================================================
                # HTTP 200
                # =================================================

                if response.status_code == 200:

                    try:

                        return response.json()

                    except ValueError as e:

                        logger.warning(
                            f"IEEE JSON decode failed: {e}"
                        )

                        return None

                # =================================================
                # 限流
                #
                # 418
                # 429
                # 403 + Service Over Qps
                # =================================================

                response_text = response.text[:1000]

                is_rate_limited = (
                    response.status_code in {
                        418,
                        429,
                    }
                    or
                    (
                        response.status_code == 403
                        and
                        "Service Over Qps" in response.text
                    )
                )

                if is_rate_limited:

                    logger.warning(
                        f"IEEE rate limit detected: "
                        f"HTTP {response.status_code}"
                    )

                    logger.warning(
                        f"IEEE response: "
                        f"{response_text[:500]}"
                    )

                    # 已经没有重试次数
                    if attempt >= self.max_retries:

                        logger.error(
                            "IEEE rate limit retries exhausted."
                        )

                        return None

                    wait_time = self.retry_backoff[
                        min(
                            attempt,
                            len(self.retry_backoff) - 1
                        )
                    ]

                    logger.warning(
                        f"Waiting {wait_time}s "
                        f"before IEEE retry..."
                    )

                    time.sleep(wait_time)

                    continue

                # =================================================
                # 其他 HTTP 错误
                # =================================================

                logger.warning(
                    f"IEEE request failed: "
                    f"HTTP {response.status_code}"
                )

                logger.warning(
                    f"IEEE response: "
                    f"{response_text[:500]}"
                )

                return None

            # =====================================================
            # 网络异常
            # =====================================================

            except requests.RequestException as e:

                self._last_request_time = time.monotonic()

                logger.warning(
                    f"IEEE request exception: {e}"
                )

                if attempt >= self.max_retries:

                    logger.error(
                        "IEEE network retries exhausted."
                    )

                    return None

                wait_time = self.retry_backoff[
                    min(
                        attempt,
                        len(self.retry_backoff) - 1
                    )
                ]

                logger.warning(
                    f"Waiting {wait_time}s "
                    f"before network retry..."
                )

                time.sleep(wait_time)

        return None

    # =============================================================
    # IEEE 原始数据 → Paper
    # =============================================================

    def convert_to_paper(
        self,
        raw_paper
    ):

        title = raw_paper.get(
            "title",
            ""
        )

        if not title:
            return None

        # =========================================================
        # 作者
        # =========================================================

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

        # =========================================================
        # 摘要
        # =========================================================

        abstract = raw_paper.get(
            "abstract",
            ""
        )

        # =========================================================
        # IEEE 页面
        # =========================================================

        url = raw_paper.get(
            "html_url",
            ""
        )

        if not url:

            url = raw_paper.get(
                "abstract_url",
                ""
            )

        # =========================================================
        # PDF
        # =========================================================

        pdf_url = raw_paper.get(
            "pdf_url"
        )

        # =========================================================
        # 研究方向
        # =========================================================

        research_direction = raw_paper.get(
            "_research_direction",
            ""
        )

        # =========================================================
        # 期刊 / 会议名称
        # =========================================================

        journal = raw_paper.get(
            "publication_title",
            ""
        )

        # =========================================================
        # 出版年份
        # =========================================================

        publication_year = raw_paper.get(
            "publication_year"
        )

        try:

            publication_year = int(
                publication_year
            )

        except (
            TypeError,
            ValueError
        ):

            publication_year = None

        # =========================================================
        # Paper
        # =========================================================

        return Paper(
            source=self.name,
            title=title,
            authors=authors,
            abstract=abstract,
            url=url,
            pdf_url=pdf_url,
            full_text=None,

            research_direction=research_direction,
            journal=journal,
            publication_year=publication_year,
        )