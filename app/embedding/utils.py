from typing import BinaryIO, List, Dict, Any
import asyncio
import os
import json
from functools import cache
from pathlib import Path

from pdfminer.high_level import extract_text as extract_pdf_text
import docx2txt
from langchain_core.documents.base import Document
from langchain_core.embeddings import Embeddings

from transformers import AutoTokenizer

from app.config import get_settings


settings = get_settings()

async def extract_text_from_pdf(file: BinaryIO) -> Document:
    content = await asyncio.to_thread(extract_pdf_text, file)
    return Document(page_content=content)


async def extract_text_from_doc(file: BinaryIO):
    content = await asyncio.to_thread(docx2txt.process, file)
    return Document(page_content=content)

async def extract_text_from_txt(file: BinaryIO):
    def read_txt(file: BinaryIO):
        return file.read().decode()
    content = await asyncio.to_thread(read_txt, file)
    return Document(page_content=content)


def _normalize_embeddings_response(embeddings):
    """
    Normalize Cohere embedding response format.

    Cohere Embed v3 returns: {"float": [[...], ...]}
    Older versions return:   [[...], ...]

    This normalizes both to: [[...], ...]
    """
    if isinstance(embeddings, dict):
        return embeddings.get("float") or next(iter(embeddings.values()))
    return embeddings


class PatchedBedrockEmbeddings:
    """Placeholder — only instantiated when provider==aws-bedrock."""

    def __init__(self, *args, **kwargs):
        try:
            from langchain_aws import BedrockEmbeddings
            from langchain_aws.embeddings.bedrock import _batch_cohere_embedding_texts as _bce
            self._bce = _bce
            self._inner = BedrockEmbeddings(*args, **kwargs)
        except ImportError as exc:
            raise ImportError(
                "aws-bedrock embedding provider requires langchain-aws. "
                "Install with: pip install langchain-aws"
            ) from exc

    def _embedding_func(self, text: str, input_type: str = "search_document") -> List[float]:
        text = text.replace(os.linesep, " ")
        if self._inner._inferred_provider == "cohere":
            response_body = self._inner._invoke_model(
                input_body={"input_type": input_type, "texts": [text]}
            )
            embeddings = _normalize_embeddings_response(response_body.get("embeddings"))
            return embeddings[0]
        else:
            response_body = self._inner._invoke_model(input_body={"inputText": text})
            return response_body.get("embedding")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self._inner.embed_documents(texts)

    def embed_query(self, text: str) -> List[float]:
        return self._inner.embed_query(text)


def get_embedder() -> Embeddings:
    """
    Returns a cached Embeddings instance based on settings.EMBED_PROVIDER.

    - "ollama"     : OllamaEmbeddings (nomic-embed-text via 10.79.85.35)
    - "aws-bedrock": PatchedBedrockEmbeddings (Cohere via Bedrock)
    """
    if settings.EMBED_PROVIDER == "ollama":
        try:
            from langchain_ollama import OllamaEmbeddings
        except ImportError as exc:
            raise ImportError(
                "ollama embedding provider requires langchain-ollama. "
                "Install with: pip install langchain-ollama"
            ) from exc
        import logging
        logging.getLogger(__name__).info(
            f"Creating OllamaEmbeddings: model={settings.EMBED_MODEL!r} "
            f"base_url={settings.EMBED_API_BASE!r}"
        )
        return OllamaEmbeddings(
            model=settings.EMBED_MODEL,
            base_url=settings.EMBED_API_BASE,
        )

    elif settings.EMBED_PROVIDER == "aws-bedrock":
        EMBED_ARN = "arn:aws:bedrock:us-west-2:233532778289:application-inference-profile/4xgakngy389z"
        embed_model = settings.EMBED_MODEL
        if not embed_model.startswith("arn:aws:bedrock:"):
            import logging as _log
            _log.getLogger(__name__).warning(
                f"EMBED_MODEL {embed_model!r} is a direct ID; using embedding profile ARN instead."
            )
            embed_model = EMBED_ARN

        embed_provider = "cohere"
        if settings.EMBED_TOKENIZER and "cohere" in settings.EMBED_TOKENIZER.lower():
            embed_provider = "cohere"
        elif settings.EMBED_TOKENIZER and "amazon" in settings.EMBED_TOKENIZER.lower():
            embed_provider = "amazon"

        return PatchedBedrockEmbeddings(
            model_id=embed_model,
            region_name=settings.AWS_REGION,
            provider=embed_provider,
        )

    else:
        raise NotImplementedError(
            f"Embedding provider {settings.EMBED_PROVIDER!r} is not supported. "
            "Supported: 'ollama', 'aws-bedrock'."
        )


@cache
def get_embedding_tokenizer():
    """Returns a cached HF tokenizer for the embeddings."""
    return AutoTokenizer.from_pretrained(settings.EMBED_TOKENIZER)


@cache
def get_prompt(name: str):
    current_dir = Path(__file__).parent
    prompt_path = current_dir / "prompts" / f"{name}_prompt.txt"
    with open(prompt_path) as f:
        prompt = f.read()
    return prompt
