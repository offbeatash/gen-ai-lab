"""
Build the FAISS vector index from data/*.txt.

Usage:
    uv run python src/agenticai/ingest.py
"""

import glob
import os

from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_tavily import TavilySearch 

load_dotenv()

# --------------------------------------------------
# Paths
# --------------------------------------------------

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

DOCS_DIR = os.path.join(PROJECT_ROOT, "data")
INDEX_DIR = os.path.join(PROJECT_ROOT, "faiss_index")

# --------------------------------------------------
# Chunk configuration
# --------------------------------------------------

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50


def load_and_split() -> list[Document]:
    """Load .txt files and split them into overlapping chunks."""

    files = glob.glob(os.path.join(DOCS_DIR, "*.txt"))

    print(f"Looking for documents in: {DOCS_DIR}")
    print(f"Found {len(files)} .txt files")

    if not files:
        raise FileNotFoundError(
            f"No .txt files found in: {DOCS_DIR}"
        )

    documents = []

    for path in files:
        with open(path, "r", encoding="utf-8") as f:
            text = f.read().strip()

        if not text:
            print(f"Skipping empty file: {path}")
            continue

        documents.append(
            Document(
                page_content=text,
                metadata={
                    "source": os.path.basename(path)
                },
            )
        )

    if not documents:
        raise ValueError("All input files are empty.")

    # Proper chunking with overlap
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    chunks = splitter.split_documents(documents)

    return chunks


def main():
    print("Loading and splitting documents...")

    chunks = load_and_split()

    print(f"Split into {len(chunks)} chunks")

    if not chunks:
        raise ValueError(
            "No chunks were created. Check your documents."
        )

    print("Embedding and building FAISS index...")

    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001"
    )

    vectorstore = FAISS.from_documents(
        documents=chunks,
        embedding=embeddings,
    )

    os.makedirs(INDEX_DIR, exist_ok=True)

    vectorstore.save_local(INDEX_DIR)

    print(f"Done! Index saved to: {INDEX_DIR}")
    print(f"Total chunks indexed: {len(chunks)}")


if __name__ == "__main__":
    main()