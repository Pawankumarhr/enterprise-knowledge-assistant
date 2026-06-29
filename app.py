"""Streamlit UI for the enterprise knowledge assistant."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

import streamlit as st
from dotenv import load_dotenv

from ingest import DATA_DIR, CHROMA_DIR, collect_data_directory_pdfs, ingest_pdfs
from rag import answer_with_sources
from utils import (
	EMPTY_CONTEXT_MESSAGE,
	ensure_directory,
	get_logger,
	normalize_error,
	validate_api_key,
	validate_question,
)

logger = get_logger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent


def _add_ui_styles() -> None:
	"""Remove Streamlit's extra top whitespace."""

	st.markdown(
		"""
		<style>
		header[data-testid="stHeader"], footer, #MainMenu {
			display: none;
		}

		.block-container {
			padding-top: 0;
			padding-bottom: 1rem;
		}

		[data-testid="stVerticalBlock"] {
			gap: 0.75rem;
		}
		</style>
		""",
		unsafe_allow_html=True,
	)


def _load_environment() -> str:
	"""Load environment variables and validate the Gemini API key."""

	load_dotenv()
	api_key = os.getenv("GOOGLE_API_KEY")
	return validate_api_key(api_key)


def _ensure_runtime_directories() -> None:
	"""Create runtime directories used by uploads and Chroma persistence."""

	ensure_directory(DATA_DIR)
	ensure_directory(CHROMA_DIR)


def _save_uploaded_files(uploaded_files: Iterable[st.runtime.uploaded_file_manager.UploadedFile]) -> list[Path]:
	"""Persist uploaded PDFs to the shared data directory."""

	saved_paths: list[Path] = []
	ensure_directory(DATA_DIR)

	for uploaded_file in uploaded_files:
		destination = DATA_DIR / uploaded_file.name
		with destination.open("wb") as file_handle:
			file_handle.write(uploaded_file.getbuffer())
		saved_paths.append(destination)

	return saved_paths


def _render_sources(sources: list[dict[str, object]]) -> None:
	"""Display source metadata and snippets for a response."""

	if not sources:
		st.info("No supporting sources were returned.")
		return

	for index, source in enumerate(sources, start=1):
		source_name = str(source.get("source", "Unknown source"))
		page = source.get("page", "Unknown page")
		snippet = str(source.get("snippet", "")).strip()
		distance = source.get("distance")
		label = f"{index}. {source_name} - page {page}"
		if distance is not None:
			label = f"{label} (distance: {distance:.4f})"

		with st.expander(label, expanded=False):
			st.write(snippet or "No snippet available.")


def _render_sidebar() -> str:
	"""Render the sidebar controls and return the validated API key."""

	with st.sidebar:
		st.title("Enterprise Knowledge Assistant")
		st.caption("Local embeddings, persistent vector search, Gemini generation.")

		api_key = ""
		try:
			api_key = _load_environment()
			st.success("Google API key loaded")
		except Exception as error:
			st.error(normalize_error(error, "API key validation failed"))

		st.divider()
		st.subheader("Documents")
		current_docs = collect_data_directory_pdfs()
		st.write(f"Indexed PDFs found in data/: {len(current_docs)}")
		if current_docs:
			for pdf_path in current_docs:
				st.write(f"- {pdf_path.name}")
		else:
			st.caption("Drop PDFs into data/ or upload them below.")

		st.divider()
		st.subheader("Upload PDFs")
		uploaded_files = st.file_uploader(
			"Upload one or more PDFs",
			type=["pdf"],
			accept_multiple_files=True,
			key="pdf_uploader",
		)

		if st.button("Process uploaded PDFs", use_container_width=True, disabled=not uploaded_files):
			try:
				saved_paths = _save_uploaded_files(uploaded_files or [])
				result = ingest_pdfs(saved_paths)
				st.success(
					f"Indexed {result.documents_indexed} PDF(s) and {result.chunks_indexed} chunk(s)."
				)
				st.rerun()
			except Exception as error:
				st.error(normalize_error(error, "Failed to ingest uploaded PDFs"))

		if st.button("Reindex PDFs in data/", use_container_width=True):
			try:
				result = ingest_pdfs()
				st.success(
					f"Reindexed {result.documents_indexed} PDF(s) and {result.chunks_indexed} chunk(s)."
				)
				st.rerun()
			except Exception as error:
				st.error(normalize_error(error, "Failed to ingest PDFs from data/"))

	return api_key


def _render_header() -> None:
	"""Render the main page header and guidance."""

	st.title("Enterprise Knowledge Assistant")
	st.write(
		"Ask questions over uploaded PDFs with local embeddings and ChromaDB-backed retrieval."
	)
	st.caption(
		"If no relevant context is found, the assistant returns: "
		f'"{EMPTY_CONTEXT_MESSAGE}"'
	)


def _render_question_panel(api_key: str) -> None:
	"""Render the question input area and answer output."""

	st.subheader("Ask a question")
	question = st.text_area(
		"Question",
		placeholder="What does the document say about policy, timelines, or responsibilities?",
		height=120,
	)

	ask_disabled = not api_key.strip()
	if ask_disabled:
		st.warning("Add GOOGLE_API_KEY in .env before asking questions.")

	if st.button("Get answer", type="primary", use_container_width=True, disabled=ask_disabled):
		try:
			normalized_question = validate_question(question)
			payload = answer_with_sources(normalized_question, api_key)
			st.session_state["last_answer"] = payload
		except Exception as error:
			st.session_state.pop("last_answer", None)
			st.error(normalize_error(error, "Failed to answer question"))

	payload = st.session_state.get("last_answer")
	if payload:
		answer = str(payload.get("answer", "")).strip()
		sources = payload.get("sources", [])

		st.subheader("Answer")
		st.write(answer or EMPTY_CONTEXT_MESSAGE)

		st.subheader("Sources")
		_render_sources(list(sources) if isinstance(sources, list) else [])


def main() -> None:
	"""Application entry point for Streamlit."""

	st.set_page_config(
		page_title="Enterprise Knowledge Assistant",
		page_icon="📄",
		layout="wide",
	)
	_ensure_runtime_directories()

	if "last_answer" not in st.session_state:
		st.session_state["last_answer"] = None

	api_key = _render_sidebar()
	_render_header()

	left_column, right_column = st.columns([1, 1])
	with left_column:
		_render_question_panel(api_key)

	with right_column:
		st.subheader("How it works")
		st.markdown(
			"""
			1. Upload PDFs here or place them in `data/`.
			2. Click `Process uploaded PDFs` or `Reindex PDFs in data/`.
			3. Ask a question.
			4. Review the answer and the source document/page references.
			"""
		)
		st.markdown(
			"The assistant only answers from retrieved document context and refuses when it cannot ground a response."
		)


if __name__ == "__main__":
	main()

