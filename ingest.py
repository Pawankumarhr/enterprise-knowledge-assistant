"""PDF ingestion pipeline for the enterprise knowledge assistant.

The ingestion flow loads PDFs from disk, extracts page-aware text with PyMuPDF,
splits the content into overlapping chunks, embeds them locally, and stores the
results in a persistent ChromaDB collection.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import fitz
from chromadb.config import Settings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

from utils import (
	EMPTY_CONTEXT_MESSAGE,
	ensure_directory,
	get_logger,
	normalize_error,
	validate_pdf_path,
)

logger = get_logger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
CHROMA_DIR = PROJECT_ROOT / "chroma_db"
COLLECTION_NAME = "enterprise_knowledge_assistant"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50


@dataclass(frozen=True)
class IngestResult:
	"""Summary for a successful ingest run."""

	documents_indexed: int
	chunks_indexed: int
	source_paths: tuple[Path, ...]


def _clean_text(text: str) -> str:
	"""Normalize whitespace in extracted PDF text."""

	return " ".join(text.split()).strip()


def _copy_into_data_dir(pdf_path: Path) -> Path:
	"""Copy an uploaded PDF into the shared data directory if needed."""

	ensure_directory(DATA_DIR)
	destination = DATA_DIR / pdf_path.name
	if pdf_path.resolve() != destination.resolve():
		shutil.copy2(pdf_path, destination)
	return destination


def load_pdf_documents(pdf_path: str | Path) -> list[Document]:
	"""Extract page-level documents from a single PDF file."""

	validated_path = validate_pdf_path(pdf_path)
	documents: list[Document] = []

	try:
		with fitz.open(validated_path) as pdf:
			if pdf.page_count == 0:
				raise ValueError(f"PDF file is empty: {validated_path.name}")

			for page_number in range(pdf.page_count):
				page = pdf.load_page(page_number)
				page_text = _clean_text(page.get_text("text"))
				if not page_text:
					continue

				documents.append(
					Document(
						page_content=page_text,
						metadata={
							"source": validated_path.name,
							"file_path": str(validated_path),
							"page": page_number + 1,
						},
					)
				)
	except Exception as error:
		raise ValueError(
			normalize_error(error, f"Failed to read PDF: {validated_path.name}")
		) from error

	if not documents:
		raise ValueError(f"No extractable text found in {validated_path.name}")
	return documents


def load_pdfs_from_paths(pdf_paths: Sequence[str | Path]) -> list[Document]:
	"""Load page-aware documents for multiple PDFs."""

	documents: list[Document] = []
	for pdf_path in pdf_paths:
		documents.extend(load_pdf_documents(pdf_path))
	return documents


def collect_data_directory_pdfs() -> list[Path]:
	"""Return all PDFs currently present in the shared data directory."""

	ensure_directory(DATA_DIR)
	return sorted(
		path for path in DATA_DIR.iterdir() if path.is_file() and path.suffix.lower() == ".pdf"
	)


def chunk_documents(documents: Sequence[Document]) -> list[Document]:
	"""Split page-aware documents into overlapping chunks."""

	splitter = RecursiveCharacterTextSplitter(
		chunk_size=CHUNK_SIZE,
		chunk_overlap=CHUNK_OVERLAP,
		separators=["\n\n", "\n", " ", ""],
	)
	return splitter.split_documents(list(documents))


def get_embedding_model() -> SentenceTransformer:
	"""Load the local sentence-transformers model used for embeddings."""

	try:
		return SentenceTransformer(EMBEDDING_MODEL_NAME)
	except Exception as error:
		raise RuntimeError(
			normalize_error(error, "Embedding model failed to load")
		) from error


def embed_texts(texts: Sequence[str]) -> list[list[float]]:
	"""Create local embeddings for a batch of texts."""

	if not texts:
		return []

	try:
		model = get_embedding_model()
		vectors = model.encode(
			list(texts),
			normalize_embeddings=True,
			show_progress_bar=False,
		)
		return vectors.tolist()
	except Exception as error:
		raise RuntimeError(normalize_error(error, "Embedding generation failed")) from error


def persist_documents(chunks: Sequence[Document]) -> None:
	"""Persist chunked documents to the ChromaDB store."""

	if not chunks:
		raise ValueError(EMPTY_CONTEXT_MESSAGE)

	try:
		import chromadb

		ensure_directory(CHROMA_DIR)
		client = chromadb.PersistentClient(
			path=str(CHROMA_DIR),
			settings=Settings(anonymized_telemetry=False),
		)
		collection = client.get_or_create_collection(name=COLLECTION_NAME)

		ids = [f"chunk-{index}" for index in range(len(chunks))]
		metadatas = [dict(chunk.metadata) for chunk in chunks]
		documents = [chunk.page_content for chunk in chunks]
		embeddings = embed_texts(documents)

		if len(embeddings) != len(documents):
			raise RuntimeError("Embedding count does not match chunk count")

		collection.upsert(
			ids=ids,
			documents=documents,
			metadatas=metadatas,
			embeddings=embeddings,
		)
	except Exception as error:
		raise RuntimeError(normalize_error(error, "ChromaDB persistence failed")) from error


def ingest_pdfs(pdf_paths: Sequence[str | Path] | None = None) -> IngestResult:
	"""Ingest PDFs from explicit paths or from the shared data directory."""

	if pdf_paths is None:
		source_paths = collect_data_directory_pdfs()
	else:
		source_paths = [validate_pdf_path(path) for path in pdf_paths]

	if not source_paths:
		raise ValueError("No documents ingested yet. Add PDFs to the data folder first.")

	copied_paths = tuple(_copy_into_data_dir(path) for path in source_paths)
	documents = load_pdfs_from_paths(copied_paths)
	if not documents:
		raise ValueError("No extractable text found in the uploaded documents.")

	chunks = chunk_documents(documents)
	persist_documents(chunks)

	logger.info("Indexed %s documents into ChromaDB", len(copied_paths))
	return IngestResult(
		documents_indexed=len(copied_paths),
		chunks_indexed=len(chunks),
		source_paths=copied_paths,
	)

