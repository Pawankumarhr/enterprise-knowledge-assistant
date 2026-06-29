# Enterprise Knowledge Assistant

Enterprise Knowledge Assistant is a local RAG application for querying uploaded PDF documents with a Streamlit UI, ChromaDB persistence, SentenceTransformers embeddings, and Gemini 2.0 Flash generation.

## Features

- Upload multiple PDFs from the UI or place them directly in `data/`.
- Extract page-aware text from PDFs with PyMuPDF.
- Chunk documents with 500-character chunks and 50-character overlap.
- Generate local embeddings with `sentence-transformers/all-MiniLM-L6-v2`.
- Store and retrieve chunks from persistent ChromaDB storage in `chroma_db/`.
- Generate grounded answers with Gemini 2.0 Flash.
- Show source document names and page numbers for answers.
- Refuse to answer when the context is not present in the uploaded documents.

## Project Structure

```text
enterprise-knowledge-assistant/
├── venv/
├── data/              ← PDFs go here
├── chroma_db/         ← auto-created by ChromaDB
├── app.py             ← Streamlit UI
├── ingest.py          ← PDF processing pipeline
├── rag.py             ← retrieval + Gemini generation
├── utils.py           ← shared helpers
├── .env               ← your API key (gitignored)
├── .env.example
├── .gitignore
├── README.md
├── e2e_test.py        ← end-to-end smoke test
└── requirements.txt
```

## Requirements

- Python 3.11 recommended
- Google AI Studio API key for Gemini
- Internet access the first time SentenceTransformers downloads the local embedding model

## Setup

1. Create and activate a virtual environment.

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

2. Install dependencies.

```powershell
pip install -r requirements.txt
```

3. Create a `.env` file in the project root.

```dotenv
GOOGLE_API_KEY=your_google_api_key_here
```

## Running the App

Start the Streamlit UI with:

```powershell
streamlit run app.py
```

## Using the Assistant

1. Add one or more PDFs through the sidebar upload control, or copy PDFs into `data/`.
2. Click `Process uploaded PDFs` or `Reindex PDFs in data/`.
3. Ask a question in the main panel.
4. Review the answer and the cited source document and page number.

If the answer is not found in the uploaded documents, the app returns:

```text
Information not found in the uploaded documents.
```

## End-to-End Smoke Test

The repository includes a simple smoke test script that creates a temporary PDF, ingests it, and verifies grounded retrieval plus the refusal path.

```powershell
.\venv\Scripts\python.exe e2e_test.py
```

## Troubleshooting

- If the app reports a missing API key, verify that `.env` contains `GOOGLE_API_KEY` and that the file is in the project root.
- If there are no documents available, add PDFs to `data/` or upload them in the UI and reindex.
- If ingestion fails on a PDF, check that the file is a valid, non-empty PDF with extractable text.
- If retrieval fails, delete `chroma_db/` and reingest the PDFs.
- The first run may take longer because the embedding model is downloaded locally.

## Implementation Notes

- `app.py` provides the UI and file upload flow.
- `ingest.py` loads PDFs, preserves page metadata, chunks text, embeds it locally, and writes to ChromaDB.
- `rag.py` retrieves context, builds the Gemini prompt, and enforces refusal behavior.
- `utils.py` contains shared validation, logging, and error formatting helpers.
