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

        # 每个关键词请求之间稍微停一下
        self.request_interval = 1.0

    # =============================================================
    # 获取 IEEE 原始论文
    # =============================================================

    def _retrieve_raw_papers(self):

        raw_papers = []

        keywords = self.retriever_config.keywords

        # 用 DOI / publication number / title 去重
        seen_ids = set()

        for keyword in keywords:

            logger.info(
                f"Searching IEEE: {keyword}"
            )

            params = {
                "apikey": self.api_key,
                "querytext": keyword,

                # 动态近五年
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

            total_records = data.get(
                "total_records",
                0
            )

            logger.info(
                f"IEEE {keyword}: "
                f"{len(articles)} papers returned, "
                f"{total_records} total matches"
            )

            for article in articles:

                # =================================================
                # 去重
                # =================================================

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

                article["_research_direction"] = keyword
		raw_papers.append(article)

            time.sleep(
                self.request_interval
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

        logger.info(
            f"IEEE total unique papers: "
            f"{len(raw_papers)}"
        )

        return raw_papers

    # =============================================================
    # IEEE API 请求
    # =============================================================

    def _request(self, params):

        try:

            response = requests.get(
                self.API_URL,
                params=params,
                timeout=self.timeout,
                headers={
                    "Accept": "application/json",
                    "User-Agent": "Mozilla/5.0",
                },
            )

            # =====================================================
            # 正常
            # =====================================================

            if response.status_code == 200:

                try:

                    return response.json()

                except ValueError as e:

                    logger.warning(
                        f"IEEE JSON decode failed: {e}"
                    )

                    return None

            # =====================================================
            # API 返回错误
            # =====================================================

            logger.warning(
                f"IEEE request failed: "
                f"HTTP {response.status_code}"
            )

            logger.warning(
                f"IEEE response: "
                f"{response.text[:500]}"
            )

            return None

        except requests.RequestException as e:

            logger.warning(
                f"IEEE request exception: {e}"
            )

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

	research_direction = raw_paper.get(
	    "_research_direction",
	    ""
	)

	journal = raw_paper.get(
   	 "publication_title",
   	 ""
	)

	publication_year = raw_paper.get(
    	"publication_year"
	)

	try:
 	   publication_year = int(publication_year)
	except (TypeError, ValueError):
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