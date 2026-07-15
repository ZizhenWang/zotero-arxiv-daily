from dataclasses import dataclass
from typing import Optional, TypeVar
from datetime import datetime
import re
import tiktoken
from openai import OpenAI
from loguru import logger
import json
RawPaperItem = TypeVar('RawPaperItem')


def _truncate_for_log(text: str | None, limit: int = 120) -> str:
    if not text:
        return ""
    compact = " ".join(text.split())
    if len(compact) <= limit:
        return compact
    return compact[:limit] + "..."


def _get_generation_kwargs(llm_params: dict) -> dict:
    generation_kwargs = llm_params.get("generation_kwargs", {})
    return dict(generation_kwargs) if generation_kwargs else {}

@dataclass
class Paper:
    source: str
    title: str
    authors: list[str]
    abstract: str
    url: str
    pdf_url: Optional[str] = None
    full_text: Optional[str] = None
    tldr: Optional[str] = None
    affiliations: Optional[list[str]] = None
    score: Optional[float] = None

    def _generate_tldr_with_llm(self, openai_client:OpenAI,llm_params:dict) -> str:
        lang = llm_params.get('language', 'English')
        generation_kwargs = _get_generation_kwargs(llm_params)
        prompt = f"Given the following information of a paper, generate a one-sentence TLDR summary in {lang}:\n\n"
        if self.title:
            prompt += f"Title:\n {self.title}\n\n"

        if self.abstract:
            prompt += f"Abstract: {self.abstract}\n\n"

        if self.full_text:
            prompt += f"Preview of main content:\n {self.full_text}\n\n"

        if not self.full_text and not self.abstract:
            logger.warning(f"Neither full text nor abstract is provided for {self.url}")
            return "Failed to generate TLDR. Neither full text nor abstract is provided"
        
        # use gpt-4o tokenizer for estimation
        enc = tiktoken.encoding_for_model("gpt-4o")
        prompt_tokens = enc.encode(prompt)
        original_prompt_tokens = len(prompt_tokens)
        prompt_tokens = prompt_tokens[:4000]  # truncate to 4000 tokens
        prompt = enc.decode(prompt_tokens)
        truncated_prompt_tokens = len(prompt_tokens)

        logger.debug(
            "Generating TLDR for {url} | model={model} | language={lang} | "
            "abstract_chars={abstract_chars} | full_text_chars={full_text_chars} | "
            "prompt_tokens={prompt_tokens} | prompt_tokens_before_truncation={original_prompt_tokens} | "
            "title={title}",
            url=self.url,
            model=generation_kwargs.get("model", "<missing>"),
            lang=lang,
            abstract_chars=len(self.abstract or ""),
            full_text_chars=len(self.full_text or ""),
            prompt_tokens=truncated_prompt_tokens,
            original_prompt_tokens=original_prompt_tokens,
            title=_truncate_for_log(self.title, 160),
        )
        
        response = openai_client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": f"You are an assistant who perfectly summarizes scientific paper, and gives the core idea of the paper to the user. Your answer should be in {lang}.",
                },
                {"role": "user", "content": prompt},
            ],
            **generation_kwargs
        )
        tldr = response.choices[0].message.content
        logger.debug(
            "Generated TLDR for {url} | response_chars={response_chars} | preview={preview}",
            url=self.url,
            response_chars=len(tldr or ""),
            preview=_truncate_for_log(tldr, 200),
        )
        return tldr
    
    def generate_tldr(self, openai_client:OpenAI,llm_params:dict) -> str:
        try:
            tldr = self._generate_tldr_with_llm(openai_client,llm_params)
            self.tldr = tldr
            return tldr
        except Exception as e:
            logger.warning(
                "Failed to generate tldr of {url}: {error_type}: {error} | "
                "base_url={base_url} | model={model} | language={language} | "
                "has_abstract={has_abstract} | has_full_text={has_full_text}",
                url=self.url,
                error_type=type(e).__name__,
                error=str(e),
                base_url=getattr(getattr(openai_client, "base_url", None), "__str__", lambda: "<unknown>")(),
                model=_get_generation_kwargs(llm_params).get("model", "<missing>"),
                language=llm_params.get("language", "English"),
                has_abstract=bool(self.abstract),
                has_full_text=bool(self.full_text),
            )
            tldr = self.abstract
            self.tldr = tldr
            return tldr

    def _generate_affiliations_with_llm(self, openai_client:OpenAI,llm_params:dict) -> Optional[list[str]]:
        if self.full_text is not None:
            generation_kwargs = _get_generation_kwargs(llm_params)
            prompt = f"Given the beginning of a paper, extract the affiliations of the authors in a python list format, which is sorted by the author order. If there is no affiliation found, return an empty list '[]':\n\n{self.full_text}"
            # use gpt-4o tokenizer for estimation
            enc = tiktoken.encoding_for_model("gpt-4o")
            prompt_tokens = enc.encode(prompt)
            original_prompt_tokens = len(prompt_tokens)
            prompt_tokens = prompt_tokens[:2000]  # truncate to 2000 tokens
            prompt = enc.decode(prompt_tokens)
            truncated_prompt_tokens = len(prompt_tokens)
            logger.debug(
                "Generating affiliations for {url} | model={model} | prompt_tokens={prompt_tokens} | "
                "prompt_tokens_before_truncation={original_prompt_tokens} | full_text_chars={full_text_chars} | title={title}",
                url=self.url,
                model=generation_kwargs.get("model", "<missing>"),
                prompt_tokens=truncated_prompt_tokens,
                original_prompt_tokens=original_prompt_tokens,
                full_text_chars=len(self.full_text or ""),
                title=_truncate_for_log(self.title, 160),
            )
            affiliations = openai_client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": "You are an assistant who perfectly extracts affiliations of authors from a paper. You should return a python list of affiliations sorted by the author order, like [\"TsingHua University\",\"Peking University\"]. If an affiliation is consisted of multi-level affiliations, like 'Department of Computer Science, TsingHua University', you should return the top-level affiliation 'TsingHua University' only. Do not contain duplicated affiliations. If there is no affiliation found, you should return an empty list [ ]. You should only return the final list of affiliations, and do not return any intermediate results.",
                    },
                    {"role": "user", "content": prompt},
                ],
                **generation_kwargs
            )
            affiliations = affiliations.choices[0].message.content

            affiliations = re.search(r'\[.*?\]', affiliations, flags=re.DOTALL).group(0)
            affiliations = json.loads(affiliations)
            affiliations = list(set(affiliations))
            affiliations = [str(a) for a in affiliations]

            logger.debug(
                "Generated affiliations for {url} | affiliation_count={affiliation_count} | preview={preview}",
                url=self.url,
                affiliation_count=len(affiliations),
                preview=_truncate_for_log(", ".join(affiliations), 200),
            )

            return affiliations
        logger.debug(
            "Skipping affiliation generation for {url} because full_text is unavailable.",
            url=self.url,
        )
    
    def generate_affiliations(self, openai_client:OpenAI,llm_params:dict) -> Optional[list[str]]:
        try:
            affiliations = self._generate_affiliations_with_llm(openai_client,llm_params)
            self.affiliations = affiliations
            return affiliations
        except Exception as e:
            logger.warning(
                "Failed to generate affiliations of {url}: {error_type}: {error} | "
                "base_url={base_url} | model={model} | has_full_text={has_full_text}",
                url=self.url,
                error_type=type(e).__name__,
                error=str(e),
                base_url=getattr(getattr(openai_client, "base_url", None), "__str__", lambda: "<unknown>")(),
                model=_get_generation_kwargs(llm_params).get("model", "<missing>"),
                has_full_text=bool(self.full_text),
            )
            self.affiliations = None
            return None
@dataclass
class CorpusPaper:
    title: str
    abstract: str
    added_date: datetime
    paths: list[str]
