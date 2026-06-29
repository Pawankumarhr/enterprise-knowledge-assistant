"""Shared utility helpers for the enterprise knowledge assistant.

This module centralizes lightweight validation and logging helpers that are
used across ingestion, retrieval, and the Streamlit UI.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Final

EMPTY_CONTEXT_MESSAGE: Final[str] = (
	"Information not found in the uploaded documents."
)


def get_logger(name: str = "enterprise_knowledge_assistant") -> logging.Logger:
	"""Return a consistently configured application logger.

	The logger is configured only once and writes plain-text messages to
	standard output. Repeated calls reuse the same logger instance.
	"""

	logger = logging.getLogger(name)
	if not logger.handlers:
		handler = logging.StreamHandler()
		handler.setFormatter(
			logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
		)
		logger.addHandler(handler)
		logger.setLevel(logging.INFO)
		logger.propagate = False
	return logger


def ensure_directory(path: str | Path) -> Path:
	"""Create a directory if needed and return it as a Path instance."""

	directory = Path(path)
	directory.mkdir(parents=True, exist_ok=True)
	return directory


def validate_api_key(api_key: str | None) -> str:
	"""Validate a Gemini API key and return the normalized value.

	Raises:
		ValueError: If the key is missing or empty.
	"""

	if api_key is None or not api_key.strip():
		raise ValueError("Missing Google API key. Set GOOGLE_API_KEY in .env.")
	return api_key.strip()


def validate_question(question: str | None) -> str:
	"""Validate a user question and return the trimmed text."""

	if question is None or not question.strip():
		raise ValueError("Please enter a question before asking the assistant.")
	return question.strip()


def validate_pdf_path(path: str | Path) -> Path:
	"""Validate that a path points to a non-empty PDF file.

	Raises:
		FileNotFoundError: If the file does not exist.
		ValueError: If the file is not a PDF or is empty.
	"""

	pdf_path = Path(path)
	if not pdf_path.exists():
		raise FileNotFoundError(f"PDF file not found: {pdf_path}")
	if pdf_path.suffix.lower() != ".pdf":
		raise ValueError(f"Unsupported file type: {pdf_path.name}")
	if pdf_path.stat().st_size <= 0:
		raise ValueError(f"PDF file is empty: {pdf_path.name}")
	return pdf_path


def normalize_error(error: Exception, context: str) -> str:
	"""Create a concise, user-facing error message for a caught exception."""

	message = str(error).strip()
	if message:
		return f"{context}: {message}"
	return context

