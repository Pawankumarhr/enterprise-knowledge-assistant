"""End-to-end smoke test for the enterprise knowledge assistant.

This script creates a temporary PDF, ingests it into an isolated Chroma store,
and verifies both the grounded answer path and the refusal behavior.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import fitz
from dotenv import load_dotenv

import ingest
import rag
from utils import EMPTY_CONTEXT_MESSAGE, validate_api_key


def _create_sample_pdf(pdf_path: Path) -> None:
    """Create a simple one-page PDF for smoke testing."""

    document = fitz.open()
    page = document.new_page()
    page.insert_text(
        (72, 72),
        "Project codename: Aurora. Deployment owner: Operations Team. "
        "Review deadline: 15 July 2026.",
        fontsize=12,
    )
    document.save(pdf_path)
    document.close()


def _configure_isolated_paths(data_dir: Path, chroma_dir: Path) -> None:
    """Point ingestion and retrieval modules at temporary directories."""

    ingest.DATA_DIR = data_dir
    ingest.CHROMA_DIR = chroma_dir
    rag.CHROMA_DIR = chroma_dir


def _assert_grounded_answer(answer: str) -> None:
    """Fail fast if the answer is not grounded in the sample PDF."""

    normalized_answer = answer.lower()
    if "aurora" not in normalized_answer:
        raise AssertionError(f"Grounded answer did not mention Aurora: {answer}")


def main() -> None:
    """Run the smoke test end to end."""

    load_dotenv()
    api_key = validate_api_key(os.getenv("GOOGLE_API_KEY"))

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as temp_root:
        temp_root_path = Path(temp_root)
        data_dir = temp_root_path / "data"
        chroma_dir = temp_root_path / "chroma_db"
        data_dir.mkdir(parents=True, exist_ok=True)
        chroma_dir.mkdir(parents=True, exist_ok=True)

        _configure_isolated_paths(data_dir=data_dir, chroma_dir=chroma_dir)

        sample_pdf = data_dir / "smoke_test.pdf"
        _create_sample_pdf(sample_pdf)

        ingest_result = ingest.ingest_pdfs([sample_pdf])
        if ingest_result.documents_indexed != 1:
            raise AssertionError(
                f"Expected one ingested PDF, got {ingest_result.documents_indexed}"
            )

        original_generate_with_gemini = rag._generate_with_gemini
        try:
            rag._generate_with_gemini = (
                lambda prompt, api_key: "The project codename is Aurora."
            )
            response = rag.answer_question(
                "What is the project codename?",
                api_key=api_key,
            )
        finally:
            rag._generate_with_gemini = original_generate_with_gemini
        if response.answer == EMPTY_CONTEXT_MESSAGE:
            raise AssertionError("Expected a grounded answer, but received refusal.")
        if not response.sources:
            raise AssertionError("Expected at least one source for the grounded answer.")
        _assert_grounded_answer(response.answer)
        first_source = response.sources[0]
        if first_source.source != sample_pdf.name:
            raise AssertionError(
                f"Expected source {sample_pdf.name}, got {first_source.source}"
            )
        if first_source.page != 1:
            raise AssertionError(f"Expected page 1, got {first_source.page}")

        original_retrieve_chunks = rag.retrieve_chunks
        try:
            rag.retrieve_chunks = lambda question, top_k=rag.DEFAULT_TOP_K: []
            refusal_response = rag.answer_question(
                "What is the budget amount?",
                api_key=api_key,
            )
        finally:
            rag.retrieve_chunks = original_retrieve_chunks

        if refusal_response.answer != EMPTY_CONTEXT_MESSAGE:
            raise AssertionError(
                "Expected refusal when no context is available, "
                f"got: {refusal_response.answer}"
            )

        print("e2e_test_passed")


if __name__ == "__main__":
    main()
