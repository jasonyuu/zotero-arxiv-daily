from .base import BaseRetriever, register_retriever
from ..protocol import Paper
from loguru import logger
import requests
import os


@register_retriever("ieee")
class IEEERetriever(BaseRetriever):

    def __init__(self, config):
        super().__init__(config)

        self.api_key = os.getenv("IEEE_API_KEY")

        if not self.api_key:
            raise ValueError(
                "IEEE_API_KEY environment variable is missing"
            )


    def _retrieve_raw_papers(self):

        raw_papers = []

        keywords = self.retriever_config.keywords


        for keyword in keywords:

            logger.info(
                f"Searching IEEE: {keyword}"
            )


            params = {

                "apikey":
                    self.api_key,

                "querytext":
                    keyword,

                "max_records":
                    10,

                "sort_field":
                    "publication_year",

                "sort_order":
                    "desc"

            }


            try:

                response = requests.get(
                    "https://ieeexploreapi.ieee.org/api/v1/search/articles",
                    params=params,
                    timeout=30
                )

                response.raise_for_status()


                data = response.json()


                articles = data.get(
                    "articles",
                    []
                )


                logger.info(
                    f"IEEE {keyword}: {len(articles)} papers"
                )


                raw_papers.extend(
                    articles
                )


            except Exception as e:

                logger.warning(
                    f"IEEE request failed: {e}"
                )


        return raw_papers



    def convert_to_paper(self, raw_paper):

        title = raw_paper.get(
            "title",
            ""
        )


        if not title:
            return None



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



        abstract = raw_paper.get(
            "abstract",
            ""
        )


        url = raw_paper.get(
            "html_url",
            ""
        )


        return Paper(

            source=self.name,

            title=title,

            authors=authors,

            abstract=abstract,

            url=url,

            pdf_url=None,

            full_text=None

        )