from dataclasses import dataclass
from typing import Optional, TypeVar
from datetime import datetime
import re
import tiktoken
from openai import OpenAI
from loguru import logger
import json


RawPaperItem = TypeVar('RawPaperItem')


def _request_llm(
    openai_client: OpenAI,
    llm_params: dict,
    messages: list[dict]
) -> str:

    api_mode = llm_params.get(
        "api_mode",
        "chat_completion"
    )

    generation_kwargs = dict(
        llm_params.get(
            "generation_kwargs",
            {}
        )
    )

    if api_mode == "chat_completion":

        response = openai_client.chat.completions.create(
            messages=messages,
            **generation_kwargs,
        )

        return response.choices[0].message.content

    if api_mode == "response":

        max_tokens = generation_kwargs.pop(
            "max_tokens",
            None
        )

        if (
            max_tokens is not None
            and "max_output_tokens"
            not in generation_kwargs
        ):
            generation_kwargs[
                "max_output_tokens"
            ] = max_tokens

        response = openai_client.responses.create(
            input=messages,
            **generation_kwargs,
        )

        return response.output_text

    raise ValueError(
        f"Unsupported llm.api_mode: {api_mode}. "
        "Expected 'chat_completion' or 'response'."
    )


@dataclass
class Paper:

    source: str
    title: str
    authors: list[str]
    abstract: str
    url: str

    pdf_url: Optional[str] = None
    full_text: Optional[str] = None

    # =========================================================
    # LLM generated content
    # =========================================================

    tldr: Optional[str] = None
    affiliations: Optional[list[str]] = None

    # =========================================================
    # Ranking
    # =========================================================

    score: Optional[float] = None

    # =========================================================
    # IEEE information
    # =========================================================

    research_direction: Optional[str] = None
    direction_score: Optional[float] = None
    journal: Optional[str] = None
    publication_year: Optional[int] = None

    # =========================================================
    # SCI information
    # =========================================================

    sci_quartile: Optional[int] = None
    sci_category: Optional[str] = None

    # =========================================================
    # Chinese paper introduction
    # =========================================================

    def _generate_tldr_with_llm(
        self,
        openai_client: OpenAI,
        llm_params: dict
    ) -> str:

        # -----------------------------------------------------
        # 强制中文简介
        # -----------------------------------------------------

        prompt = (
            "请根据下面的学术论文信息，生成一段专业、准确、"
            "简洁的中文论文简介。\n\n"
            "要求：\n"
            "1. 使用中文。\n"
            "2. 长度约 2～4 句话。\n"
            "3. 说明论文研究的问题或对象。\n"
            "4. 说明论文采用的主要方法或技术路线。\n"
            "5. 如果摘要中提供了实验、仿真或主要结果，应简要说明。\n"
            "6. 如果摘要没有提供具体结果，不要自行编造数据。\n"
            "7. 重点突出论文对电机、电机驱动、容错、高速电机、"
            "多相电机、永磁电机或新能源汽车电驱系统等方面的研究内容，"
            "但不要强行添加摘要中没有的信息。\n"
            "8. 不要使用 Markdown。\n"
            "9. 不要添加“论文简介：”等标题，只返回正文。\n\n"
        )

        if self.title:

            prompt += (
                f"论文标题：\n"
                f"{self.title}\n\n"
            )

        if self.journal:

            prompt += (
                f"期刊：\n"
                f"{self.journal}\n\n"
            )

        if self.publication_year:

            prompt += (
                f"年份：\n"
                f"{self.publication_year}\n\n"
            )

        if self.abstract:

            prompt += (
                f"论文摘要：\n"
                f"{self.abstract}\n\n"
            )

        if self.full_text:

            prompt += (
                f"论文正文前部：\n"
                f"{self.full_text}\n\n"
            )

        if (
            not self.full_text
            and not self.abstract
        ):

            logger.warning(
                f"Neither full text nor abstract is provided "
                f"for {self.url}"
            )

            return (
                "暂无可用于生成中文简介的论文摘要或正文。"
            )

        # -----------------------------------------------------
        # Token limit
        # -----------------------------------------------------

        enc = tiktoken.encoding_for_model(
            "gpt-4o"
        )

        prompt_tokens = enc.encode(
            prompt
        )

        prompt_tokens = prompt_tokens[:4000]

        prompt = enc.decode(
            prompt_tokens
        )

        # -----------------------------------------------------
        # LLM
        # -----------------------------------------------------

        tldr = _request_llm(
            openai_client,
            llm_params,
            [
                {
                    "role": "system",
                    "content": (
                        "你是一名专业的电机与电力电子领域"
                        "科研论文助手。"
                        "你的任务是准确总结学术论文。"
                        "不得编造论文中不存在的实验结果、"
                        "数据或结论。"
                        "请使用专业、自然、简洁的中文。"
                    ),
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
        )

        return tldr.strip()

    def generate_tldr(
        self,
        openai_client: OpenAI,
        llm_params: dict
    ) -> str:

        try:

            tldr = self._generate_tldr_with_llm(
                openai_client,
                llm_params
            )

            self.tldr = tldr

            return tldr

        except Exception as e:

            logger.warning(
                f"Failed to generate Chinese summary "
                f"of {self.url}: {e}"
            )

            # LLM 失败时直接使用英文摘要，
            # 至少保证邮件中仍然有论文内容。

            tldr = self.abstract

            self.tldr = tldr

            return tldr

    # =========================================================
    # Affiliations
    # =========================================================

    def _generate_affiliations_with_llm(
        self,
        openai_client: OpenAI,
        llm_params: dict
    ) -> Optional[list[str]]:

        if self.full_text is not None:

            prompt = (
                "Given the beginning of a paper, extract the "
                "affiliations of the authors in a python list "
                "format, which is sorted by the author order. "
                "If there is no affiliation found, return an "
                "empty list '[]':\n\n"
                f"{self.full_text}"
            )

            enc = tiktoken.encoding_for_model(
                "gpt-4o"
            )

            prompt_tokens = enc.encode(
                prompt
            )

            prompt_tokens = prompt_tokens[:2000]

            prompt = enc.decode(
                prompt_tokens
            )

            affiliations = _request_llm(
                openai_client,
                llm_params,
                [
                    {
                        "role": "system",
                        "content": (
                            "You are an assistant who perfectly "
                            "extracts affiliations of authors "
                            "from a paper. "
                            "You should return a python list "
                            "of affiliations sorted by the "
                            "author order, like "
                            "[\"TsingHua University\","
                            "\"Peking University\"]. "
                            "If an affiliation is consisted "
                            "of multi-level affiliations, "
                            "return the top-level affiliation "
                            "only. "
                            "Do not contain duplicated "
                            "affiliations. "
                            "If there is no affiliation found, "
                            "return an empty list [ ]. "
                            "You should only return the final "
                            "list of affiliations."
                        ),
                    },
                    {
                        "role": "user",
                        "content": prompt,
                    },
                ],
            )

            match = re.search(
                r'\[.*?\]',
                affiliations,
                flags=re.DOTALL
            )

            if not match:

                return []

            affiliations = match.group(0)

            affiliations = json.loads(
                affiliations
            )

            # 保持顺序去重
            affiliations = list(
                dict.fromkeys(
                    str(a)
                    for a in affiliations
                )
            )

            return affiliations

        return None

    def generate_affiliations(
        self,
        openai_client: OpenAI,
        llm_params: dict
    ) -> Optional[list[str]]:

        try:

            affiliations = (
                self._generate_affiliations_with_llm(
                    openai_client,
                    llm_params
                )
            )

            self.affiliations = affiliations

            return affiliations

        except Exception as e:

            logger.warning(
                f"Failed to generate affiliations "
                f"of {self.url}: {e}"
            )

            self.affiliations = None

            return None


@dataclass
class CorpusPaper:

    title: str
    abstract: str
    added_date: datetime
    paths: list[str]