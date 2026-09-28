import os
import time
from urllib.parse import quote

import requests
from loguru import logger


class EasyScholarRanker:

    API_URL = (
        "https://www.easyscholar.cc/"
        "open/getPublicationRank"
    )

    def __init__(self):

        self.secret_key = os.getenv(
            "EASY_SCHOLAR_SECRET_KEY"
        )

        if not self.secret_key:
            raise ValueError(
                "EASY_SCHOLAR_SECRET_KEY environment variable is missing"
            )

        # EasyScholar 要求每秒最多 2 次请求
        self.request_interval = 0.55

        self.timeout = 20

        # 期刊名称 -> 查询结果缓存
        self.cache = {}

    # ============================================================
    # 查询单个期刊
    # ============================================================

    def get_rank(self, publication_name):

        if not publication_name:
            return None

        publication_name = publication_name.strip()

        if not publication_name:
            return None

        # --------------------------------------------------------
        # 内存缓存
        # --------------------------------------------------------

        if publication_name in self.cache:
            return self.cache[publication_name]

        params = {
            "secretKey": self.secret_key,
            "publicationName": publication_name,
        }

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

            if response.status_code != 200:

                logger.warning(
                    f"EasyScholar request failed: "
                    f"HTTP {response.status_code}, "
                    f"journal={publication_name}"
                )

                return None

            try:

                result = response.json()

            except ValueError as e:

                logger.warning(
                    f"EasyScholar JSON decode failed: "
                    f"{publication_name}: {e}"
                )

                return None

            code = result.get("code")

            if code != 200:

                logger.warning(
                    f"EasyScholar query failed: "
                    f"journal={publication_name}, "
                    f"code={code}, "
                    f"msg={result.get('msg')}"
                )

                return None

            data = result.get(
                "data",
                {}
            )

            official_rank = data.get(
                "officialRank",
                {}
            )

            all_rank = official_rank.get(
                "all",
                {}
            )

            # ----------------------------------------------------
            # SCI-JCR
            # ----------------------------------------------------

            sci_rank = all_rank.get(
                "sci"
            )

            quartile = self.parse_sci_quartile(
                sci_rank
            )

            result_data = {
                "journal": publication_name,
                "sci": sci_rank,
                "quartile": quartile,
                "official_rank": official_rank,
                "raw": result,
            }

            self.cache[
                publication_name
            ] = result_data

            logger.info(
                f"EasyScholar: "
                f"{publication_name} -> "
                f"SCI={sci_rank}, "
                f"Q{quartile if quartile else '?'}"
            )

            return result_data

        except requests.RequestException as e:

            logger.warning(
                f"EasyScholar request exception: "
                f"{publication_name}: {e}"
            )

            return None

        finally:

            # ----------------------------------------------------
            # 限速：每秒最多2次
            # ----------------------------------------------------

            time.sleep(
                self.request_interval
            )

    # ============================================================
    # 解析 SCI 分区
    #
    # EasyScholar 常见返回：
    #
    # "Q1"
    # "Q2"
    # "Q3"
    # "Q4"
    #
    # 也兼容：
    # 1 / 2 / 3 / 4
    # ============================================================

    @staticmethod
    def parse_sci_quartile(value):

        if value is None:
            return None

        value = str(value).strip().upper()

        if value in {
            "Q1",
            "Q2",
            "Q3",
            "Q4",
        }:

            return int(
                value[1]
            )

        if value in {
            "1",
            "2",
            "3",
            "4",
        }:

            return int(value)

        return None

    # ============================================================
    # 给论文添加 SCI 信息
    # ============================================================

    def enrich_paper(self, paper):

        journal = getattr(
            paper,
            "journal",
            None
        )

        if not journal:

            logger.warning(
                f"No journal name for paper: "
                f"{paper.title}"
            )

            return paper

        rank = self.get_rank(
            journal
        )

        if rank is None:

            paper.sci_quartile = None
            paper.sci_category = None

            return paper

        paper.sci_quartile = (
            rank["quartile"]
        )

        paper.sci_category = (
            rank["sci"]
        )

        return paper