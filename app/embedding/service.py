"""
app/embedding/service.py  (REFACTORED)

Changes from original:
  - save_embedding_to_s3 -> uses storage_upload_to_prefix
  - load_embedding_from_s3 -> uses storage_download_fileobj
  - Function names kept the same (they still say "s3" for backward compat with callers)
    but internally route through the storage abstraction layer.
"""

from typing import BinaryIO, Optional
import asyncio
import logging
import pickle
from tempfile import SpooledTemporaryFile

from langchain_core.documents.base import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage
from langchain_core.messages.utils import count_tokens_approximately
from langchain.text_splitter import RecursiveCharacterTextSplitter

from app.config import get_settings
from app.core.llm import get_model
from app.core.utils import extract_lc_msg_content
from app.background_mgr.service import TaskProgressTracker
from app.storage import storage_upload_to_prefix, storage_download_fileobj
from app.agent.utils import CACHE_POINT_BLOCK

from .schemas import EmbeddedPassage, EmbeddedDocument
from .utils import (
    extract_text_from_doc,
    extract_text_from_pdf,
    extract_text_from_txt,
    get_embedding_tokenizer,
    get_embedder,
    get_prompt
)
from .exceptions import FileNotSupportedError, EmbeddingException

settings = get_settings()
logger = logging.getLogger(__name__)

async def _embedding_error_handler(task_id: str, exc: Exception, bg_tracker: Optional[TaskProgressTracker] = None):
    err_str = f"Error embedding document for task {task_id}: {str(exc)}"
    logger.error(err_str)
    if bg_tracker:
        await bg_tracker.fail(err_str)
    raise exc

async def embed_document(
    task_id: str,
    file: BinaryIO,
    filename: str,
    filetype: str,
    output_loc: str,
    vault_key: str = "",
    bg_tracker: Optional[TaskProgressTracker] = None,
):
    """Embed a document with context

    task_id: the id for this embedding task
    file: A file-like object containing the document
    filetype: MIME type of the document
    output_loc: The storage prefix for saving embedding outputs
    bg_tracker: Optional progress bg_tracker for progress logging
    """
    logger.info(f"Starting document embedding for task {task_id}")

    try:
        # Extract text and check for length
        text_doc = await extract_text(file, filetype)
        text_doc.metadata["filename"] = filename
        max_size = (settings.ELLM_CTX_LEN - settings.EMBED_CHUNK_SIZE) * 0.85
        if (tk_cnt_est := count_tokens_approximately([text_doc.page_content])) > max_size:
            raise EmbeddingException("Document too long")

        # Chunk the text
        chunks = await chunk_text(text_doc)
        if bg_tracker:
            await bg_tracker.start(len(chunks) + int(len(chunks) / settings.EMBED_BATCH_SIZE) + 4, "Text extracted")
            await bg_tracker.increment(1, "Embedding started")

        # Generate summary, keyword, inject context, and embed
        summary, keywords, chunks = await generate_context(task_id, text_doc, chunks, bg_tracker)
        embeddings = await embed_chunks(task_id, chunks, bg_tracker)

        # Pack and upload result
        result = EmbeddedDocument(
            name=filename,
            est_size=tk_cnt_est,
            summary=summary,
            keywords=keywords,
            passages=[
                EmbeddedPassage(
                    text=chunk.page_content,
                    embedding=embedding,
                    metadata=chunk.metadata
                )
                for chunk, embedding in zip(chunks, embeddings)
            ]
        )
        await save_embedding_to_s3(task_id, result, output_loc)

        if bg_tracker:
            await bg_tracker.complete(message="Embedding successfully completed and saved.")

    except Exception as e:
       await _embedding_error_handler(task_id, e, bg_tracker)

async def extract_text(file: BinaryIO, filetype: str) -> Document:
    match filetype:
        case "application/pdf":
            return await extract_text_from_pdf(file)
        case "text/plain":
            return await extract_text_from_txt(file)
        case "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            return await extract_text_from_doc(file)
        case _:
            raise FileNotSupportedError(f"File type {filetype} is not supported")

async def chunk_text(doc: Document) -> list[Document]:
    logger.info("Chunking document")
    text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        tokenizer=get_embedding_tokenizer(),
        chunk_size=settings.EMBED_CHUNK_SIZE,
        chunk_overlap=settings.EMBED_OVERLAP,
        separators=["\n\n", "\n", ".", "\u3002", "?", "!", " ", ""]
    )
    return list(await text_splitter.atransform_documents(documents=[doc]))

async def generate_context(
    task_id: str,
    document: Document,
    chunks: list[Document],
    bg_tracker: Optional[TaskProgressTracker] = None
) -> tuple[str, list[str], list[Document]]:
    """Generate document summary, keywords, and inject context to chunks."""
    logger.info(f"Generating context for {len(chunks)} chunks for task {task_id}")

    doc_context = f"<document>\n{document.page_content}\n</document>"
    doc_w_cache = [
        {"type": "text", "text": doc_context},
        CACHE_POINT_BLOCK
    ]
    summary_prompt = get_prompt("summarize").format(length=settings.SUMMARY_LEN)
    keyword_prompt = get_prompt("keyword")
    context_prompt = get_prompt("context")
    context_prompts = [context_prompt.format(chunk=chunk.page_content) for chunk in chunks]

    final_prompts = [
        HumanMessage(content=[
            *doc_w_cache,
            {"type": "text", "text": summary_prompt}
        ]),
        HumanMessage(content=[
            *doc_w_cache,
            {"type": "text", "text": keyword_prompt}
        ]),
        *[
            HumanMessage(content=[
                *doc_w_cache,
                {"type": "text", "text": context_prompt}
            ]) for context_prompt in context_prompts
        ]
    ]

    responses = []
    llm = get_model(efficient=True)
    if bg_tracker:
        tasks = [
            bg_tracker.with_tracker(
                llm.ainvoke([prompt]),
                message="Generating contexts..."
            ) for prompt in final_prompts
        ]
    else:
        tasks = [llm.ainvoke([prompt]) for prompt in final_prompts]

    # Initialize cache point by waiting for the first response
    responses.append((await tasks[0]))
    # Concurrently call inference on rest of prompts
    responses.extend(await asyncio.gather(*tasks[1:]))

    # Get summary and keywords
    summary = extract_lc_msg_content(responses[0])[0].get("text", "")
    keywords = list(map(str.strip, extract_lc_msg_content(responses[1])[0].get("text", "").split(",")))

    # Prepend generated contexts to each chunk
    for i, (doc, res) in enumerate(zip(chunks, responses[2:])):
        context = extract_lc_msg_content(res)[0].get("text", "")
        if not isinstance(context, str):
            raise EmbeddingException(
                f"Generated context not a string for doc {doc.metadata['name']} chunk {i}"
            )
        doc.page_content = context + "\n\n" + doc.page_content
        doc.metadata["chunk_index"] = i

    return summary, keywords, chunks

async def embed_chunks(
    task_id: str,
    chunks: list[Document],
    bg_tracker: Optional[TaskProgressTracker] = None
) -> list[list[float]]:
    """Embed chunks of text"""
    logger.info(f"Embedding {len(chunks)} chunks for task {task_id}")

    embedder = get_embedder()
    texts = [c.page_content for c in chunks]
    if bg_tracker:
        tasks = [
            bg_tracker.with_tracker(
                embedder.aembed_documents(texts[i:i+settings.EMBED_BATCH_SIZE]),
                message="Embedding passages..."
            )
            for i in range(0, len(chunks), settings.EMBED_BATCH_SIZE)
        ]
    else:
        tasks = [
            embedder.aembed_documents(texts[i:i+settings.EMBED_BATCH_SIZE])
            for i in range(0, len(chunks), settings.EMBED_BATCH_SIZE)
        ]
    embeddings = sum(await asyncio.gather(*tasks), [])

    return embeddings

async def save_embedding_to_s3(
    task_id: str,
    embedding: EmbeddedDocument,
    prefix: str
) -> None:
    """Save embedding to storage (local or S3 - transparent)."""
    with SpooledTemporaryFile(max_size=settings.SPOOLED_MAX_SIZE) as temp_file:
        pickle.dump(embedding, temp_file)
        temp_file.seek(0)
        await storage_upload_to_prefix(
            bucket=settings.AWS_FILESTORE_BUCKET,
            prefix=prefix,
            file=temp_file,
            name=f"{task_id}_embeddings.pkl",
        )

async def load_embedding_from_s3(
    task_id: str,
    prefix: str
) -> EmbeddedDocument:
    """Load embedding from storage (local or S3 - transparent)."""
    if prefix and not prefix.endswith('/'):
        prefix += '/'

    embedding_fp = await storage_download_fileobj(
        settings.AWS_FILESTORE_BUCKET,
        f"{prefix}{task_id}_embeddings.pkl",
    )

    embedding = pickle.load(embedding_fp)

    return embedding
