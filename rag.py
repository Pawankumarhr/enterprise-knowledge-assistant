"""Retrieval and generation layer for the enterprise knowledge assistant.

This module pulls relevant chunks from ChromaDB, builds a grounded prompt, and
uses Gemini 2.0 Flash to generate a response that is constrained to the
retrieved context.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import chromadb
import google.generativeai as genai
from chromadb.config import Settings

from ingest import COLLECTION_NAME, CHROMA_DIR, get_embedding_model
from utils import (
	EMPTY_CONTEXT_MESSAGE,
	get_logger,
	normalize_error,
	validate_api_key,
	validate_question,
)

logger = get_logger(__name__)

GEMINI_MODEL_NAME = "gemini-2.0-flash"
DEFAULT_TOP_K = 4
_ANSWER_STOPWORDS = {
	"what",
	"which",
	"who",
	"whom",
	"whose",
	"when",
	"where",
	"why",
	"how",
	"the",
	"a",
	"an",
	"is",
	"are",
	"was",
	"were",
	"do",
	"does",
	"did",
	"about",
	"document",
	"say",
	"tell",
	"me",
	"please",
}


@dataclass(frozen=True)
class RetrievedChunk:
	"""A single retrieved chunk with source metadata."""

	content: str
	source: str
	page: int
	distance: float | None = None


@dataclass(frozen=True)
class RagResponse:
	"""Structured answer payload returned by the RAG pipeline."""

	answer: str
	sources: tuple[RetrievedChunk, ...]


def _normalize_source_metadata(metadata: dict[str, object] | None) -> tuple[str, int]:
	"""Extract a readable source name and page number from chunk metadata."""

	metadata = metadata or {}
	source = str(metadata.get("source", "Unknown source"))
	try:
		page = int(metadata.get("page", 0) or 0)
	except (TypeError, ValueError):
		page = 0
	return source, page


def _get_chroma_collection() -> chromadb.api.models.Collection.Collection:
	"""Open the persistent Chroma collection used by the app."""

	try:
		client = chromadb.PersistentClient(
			path=str(CHROMA_DIR),
			settings=Settings(anonymized_telemetry=False),
		)
		return client.get_collection(name=COLLECTION_NAME)
	except Exception as error:
		raise ValueError(
			normalize_error(error, "No documents ingested yet. Add PDFs to the data folder first.")
		) from error


def _embed_question(question: str) -> list[float]:
	"""Create a local embedding for a single question."""

	try:
		model = get_embedding_model()
		vector = model.encode([question], normalize_embeddings=True, show_progress_bar=False)
		return vector[0].tolist()
	except Exception as error:
		raise RuntimeError(normalize_error(error, "Embedding generation failed")) from error


def retrieve_chunks(question: str, top_k: int = DEFAULT_TOP_K) -> list[RetrievedChunk]:
	"""Retrieve the most relevant chunks for a question."""

	normalized_question = validate_question(question)
	collection = _get_chroma_collection()

	if collection.count() == 0:
		raise ValueError(
			"No documents ingested yet. Add PDFs to the data folder first."
		)

	try:
		results = collection.query(
			query_embeddings=[_embed_question(normalized_question)],
			n_results=top_k,
			include=["documents", "metadatas", "distances"],
		)
	except Exception as error:
		raise RuntimeError(normalize_error(error, "ChromaDB retrieval failed")) from error

	documents = results.get("documents", [[]])[0] or []
	metadatas = results.get("metadatas", [[]])[0] or []
	distances = results.get("distances", [[]])[0] or []

	retrieved_chunks: list[RetrievedChunk] = []
	for index, content in enumerate(documents):
		source, page = _normalize_source_metadata(
			metadatas[index] if index < len(metadatas) else None
		)
		distance = distances[index] if index < len(distances) else None
		retrieved_chunks.append(
			RetrievedChunk(
				content=str(content).strip(),
				source=source,
				page=page,
				distance=float(distance) if distance is not None else None,
			)
		)

	return [chunk for chunk in retrieved_chunks if chunk.content]


def format_context(chunks: Sequence[RetrievedChunk]) -> str:
	"""Format retrieved chunks into a prompt-ready context block."""

	if not chunks:
		return ""

	formatted_blocks: list[str] = []
	for index, chunk in enumerate(chunks, start=1):
		page_text = f"page {chunk.page}" if chunk.page else "unknown page"
		formatted_blocks.append(
			f"[{index}] Source: {chunk.source} | {page_text}\n{chunk.content}"
		)
	return "\n\n".join(formatted_blocks)


def _format_conversation_history(
	conversation_history: Sequence[dict[str, str]] | None,
) -> str:
	"""Format recent Q&A turns for prompt context."""

	if not conversation_history:
		return "Conversation history: None"

	formatted_turns: list[str] = []
	for index, turn in enumerate(conversation_history, start=1):
		user_question = turn.get("question", "").strip()
		assistant_answer = turn.get("answer", "").strip()
		formatted_turns.append(
			f"Turn {index}\nQ: {user_question}\nA: {assistant_answer}"
		)
	return "Conversation history:\n" + "\n\n".join(formatted_turns)


def build_prompt(
	question: str,
	chunks: Sequence[RetrievedChunk],
	conversation_history: Sequence[dict[str, str]] | None = None,
) -> str:
	"""Build a grounded prompt that forces the model to stay on context."""

	context = format_context(chunks)
	history_text = _format_conversation_history(conversation_history)
	return (
		"You are an enterprise knowledge assistant. Answer strictly from the provided "
		"context from uploaded documents.\n\n"
		"Rules:\n"
		"- Use only the context below.\n"
		"- Use the conversation history only as prior context for follow-up questions.\n"
		f"- If the answer cannot be found in the context, reply with exactly: {EMPTY_CONTEXT_MESSAGE}\n"
		"- Do not guess, do not add external knowledge, and do not mention missing context.\n"
		"- Keep the answer concise and factual.\n\n"
		f"{history_text}\n\n"
		f"Question: {question}\n\n"
		f"Context:\n{context}\n\n"
		"Answer:"
	)


def _generate_with_gemini(prompt: str, api_key: str) -> str:
	"""Call Gemini 2.0 Flash with the supplied prompt."""

	try:
		validated_key = validate_api_key(api_key)
		genai.configure(api_key=validated_key)
		model = genai.GenerativeModel(GEMINI_MODEL_NAME)
		response = model.generate_content(prompt)
		text = getattr(response, "text", "") or ""
		return text.strip()
	except Exception as error:
		raise RuntimeError(normalize_error(error, "Gemini API failed")) from error


def _extractive_fallback_answer(question: str, chunks: Sequence[RetrievedChunk]) -> str:
	"""Build a grounded answer directly from retrieved text when Gemini is unavailable."""

	keywords = {
		word.lower()
		for word in re.findall(r"[A-Za-z][A-Za-z0-9_-]+", question)
		if word.lower() not in _ANSWER_STOPWORDS
	}
	best_sentence = ""
	best_score = 0

	for chunk in chunks:
		sentences = re.split(r"(?<=[.!?])\s+", chunk.content)
		for sentence in sentences:
			candidate = sentence.strip()
			if not candidate:
				continue

			lower_candidate = candidate.lower()
			score = sum(1 for keyword in keywords if keyword in lower_candidate)
			if score > best_score:
				best_sentence = candidate
				best_score = score

	if best_sentence:
		return best_sentence

	first_chunk = chunks[0].content.strip() if chunks else ""
	if not first_chunk:
		return EMPTY_CONTEXT_MESSAGE

	return first_chunk if len(first_chunk) <= 300 else f"{first_chunk[:297].rstrip()}..."


def clean_answer_text(answer: str) -> str:
	"""Remove prompt echoes and retrieved-context artifacts from model output."""

	text = (answer or "").strip()
	if not text:
		return ""

	lines = [line.strip() for line in text.splitlines() if line.strip()]
	filtered_lines: list[str] = []
	for line in lines:
		lower_line = line.lower()
		if lower_line.startswith("context:") or lower_line.startswith("sources:"):
			continue
		if lower_line.startswith("[") and "]" in lower_line and "source:" in lower_line:
			continue
		if lower_line.startswith("answer:"):
			line = line.split(":", 1)[1].strip()
			if not line:
				continue
		filtered_lines.append(line)

	cleaned = "\n".join(filtered_lines).strip()
	return cleaned or text


def answer_question(
	question: str,
	api_key: str,
	top_k: int = DEFAULT_TOP_K,
	conversation_history: Sequence[dict[str, str]] | None = None,
) -> RagResponse:
	"""Answer a question using retrieved document context and Gemini."""

	normalized_question = validate_question(question)
	retrieved_chunks = retrieve_chunks(normalized_question, top_k=top_k)

	if not retrieved_chunks:
		return RagResponse(answer=EMPTY_CONTEXT_MESSAGE, sources=tuple())

	prompt = build_prompt(normalized_question, retrieved_chunks, conversation_history)
	try:
		answer = _generate_with_gemini(prompt, api_key)
	except RuntimeError as error:
		message = str(error)
		if "Gemini API failed" not in message:
			raise
		answer = _extractive_fallback_answer(normalized_question, retrieved_chunks)

	answer = clean_answer_text(answer)

	if not answer or answer == EMPTY_CONTEXT_MESSAGE:
		answer = EMPTY_CONTEXT_MESSAGE

	return RagResponse(answer=answer, sources=tuple(retrieved_chunks))


def answer_with_sources(
	question: str,
	api_key: str,
	top_k: int = DEFAULT_TOP_K,
	conversation_history: Sequence[dict[str, str]] | None = None,
) -> dict[str, object]:
	"""Convenience wrapper that returns a serializable answer payload."""

	response = answer_question(
		question=question,
		api_key=api_key,
		top_k=top_k,
		conversation_history=conversation_history,
	)
	return {
		"answer": response.answer,
		"sources": [
			{
				"source": chunk.source,
				"page": chunk.page,
				"snippet": chunk.content,
				"distance": chunk.distance,
			}
			for chunk in response.sources
		],
	}

