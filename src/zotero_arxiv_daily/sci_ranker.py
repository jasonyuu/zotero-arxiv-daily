import os
import re
import time

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
                "EASY_SCHOLAR_SECRET_KEY "
                "environment variable is missing"
            )

        # =========================================================
        # EasyScholar 请求频率控制
        #
        # 官方限制：
        # 每秒最多约 2 次请求
        #
        # 0.55 秒 / 次更加保守
        # =========================================================

        self.request_interval = 0.55

        self.timeout = 20

        # =========================================================
        # journal -> result
        #
        # 避免同一期刊重复查询
        # =========================================================

        self.cache = {}

    # =============================================================
    # SCI quartile parser
    # =============================================================

    @staticmethod
    def parse_sci_quartile(value):

        if value is None:
            return None

        # ---------------------------------------------------------
        # 数字
        # ---------------------------------------------------------

        if isinstance(value, int):

            if value in (1, 2, 3, 4):
                return value

            return None

        # ---------------------------------------------------------
        # 字符串
        # ---------------------------------------------------------

        value = str(
            value
        ).strip().upper()

        if not value:
            return None

        # Q1 / Q2 / Q3 / Q4
        if value in {
            "Q1",
            "Q2",
            "Q3",
            "Q4",
        }:

            return int(
                value[1]
            )

        # 纯数字
        if value in {
            "1",
            "2",
            "3",
            "4",
        }:

            return int(
                value
            )

        # ---------------------------------------------------------
        # 常见中文格式
        #
        # 1区
        # 2区
        # 3区
        # 4区
        # ---------------------------------------------------------

        match = re.search(
            r"([1-4])\s*区",
            value
        )

        if match:

            return int(
                match.group(1)
            )

        # ---------------------------------------------------------
        # Q1区 / Q2区
        # ---------------------------------------------------------

        match = re.search(
            r"Q\s*([1-4])",
            value
        )

        if match:

            return int(
                match.group(1)
            )

        return None

    # =============================================================
    # EasyScholar query
    # =============================================================

    def get_rank(
        self,
        publication_name
    ):

        # =========================================================
        # publication name validation
        # =========================================================

        if not publication_name:

            return None

        publication_name = str(
            publication_name
        ).strip()

        if not publication_name:

            return None

        # =========================================================
        # cache
        # =========================================================

        if publication_name in self.cache:

            return self.cache[
                publication_name
            ]

        # =========================================================
        # request parameters
        # =========================================================

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
                    "User-Agent": (
                        "Mozilla/5.0 "
                        "(Windows NT 10.0; Win64; x64) "
                        "AppleWebKit/537.36 "
                        "Chrome/153.0 Safari/537.36"
                    ),
                },
            )

            # =====================================================
            # HTTP status
            # =====================================================

            if response.status_code != 200:

                logger.warning(
                    "EasyScholar request failed: "
                    f"HTTP {response.status_code}, "
                    f"journal={publication_name}"
                )

                return None

            # =====================================================
            # JSON
            # =====================================================

            try:

                result = response.json()

            except ValueError as e:

                logger.warning(
                    "EasyScholar JSON decode failed: "
                    f"journal={publication_name}, "
                    f"error={e}"
                )

                return None

            # =====================================================
            # API code
            # =====================================================

            if not isinstance(
                result,
                dict
            ):

                logger.warning(
                    "EasyScholar returned unexpected "
                    f"JSON type: "
                    f"journal={publication_name}, "
                    f"type={type(result).__name__}"
                )

                return None

            code = result.get(
                "code"
            )

            if code != 200:

                logger.warning(
                    "EasyScholar query failed: "
                    f"journal={publication_name}, "
                    f"code={code}, "
                    f"msg={result.get('msg')}"
                )

                return None

            # =====================================================
            # data
            # =====================================================

            data = result.get(
                "data"
            )

            if not isinstance(
                data,
                dict
            ):

                logger.warning(
                    "EasyScholar returned no valid data: "
                    f"journal={publication_name}"
                )

                return None

            # =====================================================
            # officialRank
            # =====================================================

            official_rank = data.get(
                "officialRank"
            )

            if not isinstance(
                official_rank,
                dict
            ):

                logger.info(
                    "EasyScholar has no officialRank: "
                    f"journal={publication_name}"
                )

                result_data = {
                    "journal": publication_name,
                    "sci": None,
                    "quartile": None,
                    "official_rank": None,
                    "raw": result,
                }

                self.cache[
                    publication_name
                ] = result_data

                return result_data

            # =====================================================
            # all
            #
            # 关键修复：
            #
            # EasyScholar 有时候：
            #
            # "all": null
            #
            # 所以不能直接：
            #
            # all_rank.get(...)
            # =====================================================

            all_rank = official_rank.get(
                "all"
            )

            if not isinstance(
                all_rank,
                dict
            ):

                logger.info(
                    "EasyScholar has no officialRank.all: "
                    f"journal={publication_name}"
                )

                result_data = {
                    "journal": publication_name,
                    "sci": None,
                    "quartile": None,
                    "official_rank": official_rank,
                    "raw": result,
                }

                self.cache[
                    publication_name
                ] = result_data

                return result_data

            # =====================================================
            # SCI
            # =====================================================

            sci_rank = all_rank.get(
                "sci"
            )

            quartile = (
                self.parse_sci_quartile(
                    sci_rank
                )
            )

            # =====================================================
            # 保存结果
            # =====================================================

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

            # =====================================================
            # logging
            # =====================================================

            if quartile is not None:

                logger.info(
                    "EasyScholar: "
                    f"{publication_name} -> "
                    f"SCI={sci_rank}, "
                    f"Q{quartile}"
                )

            else:

                logger.info(
                    "EasyScholar: "
                    f"{publication_name} -> "
                    f"SCI={sci_rank}, "
                    f"Q?"
                )

            return result_data

        # =========================================================
        # requests exception
        # =========================================================

        except requests.RequestException as e:

            logger.warning(
                "EasyScholar request exception: "
                f"journal={publication_name}, "
                f"error={e}"
            )

            return None

        # =========================================================
        # Unexpected exception
        #
        # 单篇论文失败不能让整个任务退出
        # =========================================================

        except Exception as e:

            logger.exception(
                "Unexpected EasyScholar error: "
                f"journal={publication_name}, "
                f"error={e}"
            )

            return None

        finally:

            # =====================================================
            # API rate limit
            # =====================================================

            time.sleep(
                self.request_interval
            )

    # =============================================================
    # Enrich paper
    # =============================================================

    def enrich_paper(
        self,
        paper
    ):

        journal = getattr(
            paper,
            "journal",
            None
        )

        # =========================================================
        # 没有期刊名
        # =========================================================

        if not journal:

            logger.warning(
                "No journal name for paper: "
                f"{getattr(paper, 'title', 'Unknown')}"
            )

            paper.sci_quartile = None
            paper.sci_category = None

            return paper

        # =========================================================
        # Query EasyScholar
        # =========================================================

        rank = self.get_rank(
            journal
        )

        # =========================================================
        # 查询失败
        # =========================================================

        if rank is None:

            paper.sci_quartile = None
            paper.sci_category = None

            return paper

        # =========================================================
        # 保存 SCI quartile
        # =========================================================

        paper.sci_quartile = (
            rank.get(
                "quartile"
            )
        )

        # =========================================================
        # 保存 SCI category
        # =========================================================

        paper.sci_category = (
            rank.get(
                "sci"
            )
        )

        return paper