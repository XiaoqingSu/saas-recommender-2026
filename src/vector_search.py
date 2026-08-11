"""Build and query a Chroma index over saas_cleaned.searchable_text."""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

import pandas as pd
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import CharacterTextSplitter

ROOT = Path(__file__).resolve().parents[1]
CLEANED_PATH = ROOT / "data" / "saas_cleaned.csv"
COLLECTION_NAME = "saas_products"


def _chroma_dir_for(provider: str, model: str) -> Path:
    """Keep separate Chroma dirs per embedding model (dimension-safe)."""
    safe = model.replace("/", "_").replace(" ", "_")
    return ROOT / ".chroma" / f"saas_products_{provider}_{safe}"


def resolve_chroma_dir(provider: str | None = None) -> Path:
    load_dotenv(ROOT / ".env", override=True)
    choice = (provider or os.getenv("EMBEDDING_PROVIDER", "qwen")).lower()
    if choice == "qwen":
        model = os.getenv("QWEN_EMBEDDING_MODEL", "Qwen/Qwen3-Embedding-0.6B")
    elif choice == "voyage":
        model = os.getenv("VOYAGE_EMBEDDING_MODEL", "voyage-3.5")
    elif choice == "openai":
        model = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
    else:
        model = os.getenv("LOCAL_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    return _chroma_dir_for(choice, model)


def get_chroma_dir() -> Path:
    """Resolve Chroma path from current .env (prefer over stale module-level CHROMA_DIR)."""
    return resolve_chroma_dir()


# Convenience default; prefer get_chroma_dir() after .env changes
CHROMA_DIR = resolve_chroma_dir()

# Metadata kept for filtering / display (no website_url / product_hunt_url in source)
METADATA_COLS = [
    "id",
    "name",
    "tagline",
    "votes_count",
    "comments_count",
    "topics",
    "main_category",
    "platforms",
    "log_votes",
    "engagement_ratio",
    "is_ai_product",
    "is_saas_product",
    "is_dev_tool",
    "is_productivity",
    "is_viral",
    "daily_rank_clean",
    "weekly_rank_clean",
]


def _meta_value(value: Any) -> str | int | float | bool:
    """Chroma metadata only accepts str/int/float/bool (not numpy scalars)."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    # pandas/numpy scalars → Python builtins
    if hasattr(value, "item") and not isinstance(value, (str, bytes)):
        try:
            value = value.item()
        except (ValueError, AttributeError):
            pass
    if isinstance(value, (bool, int, float, str)):
        return value
    if pd.isna(value):
        return ""
    return str(value)


def row_to_document(row: pd.Series) -> Document:
    metadata = {col: _meta_value(row[col]) for col in METADATA_COLS if col in row.index}
    return Document(page_content=str(row["searchable_text"]), metadata=metadata)


def load_cleaned(path: Path = CLEANED_PATH, sample_size: int | None = None) -> pd.DataFrame:
    df = pd.read_csv(path)
    if sample_size is not None:
        df = df.head(sample_size).copy()
    return df


def build_documents(
    df: pd.DataFrame,
    chunk_size: int = 1000,
    chunk_overlap: int = 100,
) -> list[Document]:
    """Split long searchable_text; most PH blurbs stay as one chunk."""
    docs = [row_to_document(row) for _, row in df.iterrows()]
    splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    return splitter.split_documents(docs)


def get_embeddings(provider: str | None = None) -> Embeddings:
    """
    provider:
      - "qwen": Qwen3-Embedding (default, local via sentence-transformers)
      - "voyage": VoyageAI voyage-3.5 series — needs VOYAGE_API_KEY
      - "openai": needs OPENAI_API_KEY
      - "local": sentence-transformers/all-MiniLM-L6-v2 (free)
    """
    load_dotenv(ROOT / ".env", override=True)

    # HF_TOKEN unlocks higher Hub rate limits for model downloads
    hf_token = os.getenv("HF_TOKEN")
    if hf_token:
        os.environ["HUGGING_FACE_HUB_TOKEN"] = hf_token
        os.environ["HF_TOKEN"] = hf_token

    choice = (provider or os.getenv("EMBEDDING_PROVIDER", "qwen")).lower()

    if choice == "qwen":
        from langchain_huggingface import HuggingFaceEmbeddings

        model_name = os.getenv("QWEN_EMBEDDING_MODEL", "Qwen/Qwen3-Embedding-0.6B")
        # Qwen3 is instruction-aware: use built-in "query" prompt for queries only
        return HuggingFaceEmbeddings(
            model_name=model_name,
            query_encode_kwargs={"prompt_name": "query"},
            encode_kwargs={"normalize_embeddings": True},
        )

    if choice == "voyage":
        from langchain_voyageai import VoyageAIEmbeddings

        if not os.getenv("VOYAGE_API_KEY"):
            raise ValueError("VOYAGE_API_KEY missing. Set it in .env.")
        model = os.getenv("VOYAGE_EMBEDDING_MODEL", "voyage-3.5")
        # Small batches help free-tier limits (3 RPM / 10K TPM without billing)
        batch_size = int(os.getenv("VOYAGE_BATCH_SIZE", "16"))
        return VoyageAIEmbeddings(model=model, batch_size=batch_size)

    if choice == "openai":
        from langchain_openai import OpenAIEmbeddings

        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError("OPENAI_API_KEY missing. Set it in .env or use provider='local'.")
        return OpenAIEmbeddings(model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"))

    if choice == "local":
        from langchain_huggingface import HuggingFaceEmbeddings

        model_name = os.getenv("LOCAL_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
        return HuggingFaceEmbeddings(model_name=model_name)

    raise ValueError(f"Unknown EMBEDDING_PROVIDER={choice!r}. Use qwen | voyage | openai | local.")


def build_chroma(
    documents: list[Document],
    embeddings: Embeddings | None = None,
    persist_directory: Path | None = None,
    collection_name: str = COLLECTION_NAME,
    rebuild: bool = True,
    batch_size: int | None = None,
    sleep_seconds: float | None = None,
) -> Chroma:
    """
    Build a persisted Chroma index.

    For Voyage free tier (no payment method), index in small batches with sleep
    to stay under ~3 RPM / 10K TPM. Override via VOYAGE_INDEX_BATCH_SIZE /
    VOYAGE_INDEX_SLEEP_SEC.
    """
    import shutil

    persist_directory = Path(persist_directory or get_chroma_dir())
    persist_directory.mkdir(parents=True, exist_ok=True)
    embeddings = embeddings or get_embeddings()

    if rebuild and any(persist_directory.iterdir()):
        shutil.rmtree(persist_directory, ignore_errors=True)
        persist_directory.mkdir(parents=True, exist_ok=True)

    load_dotenv(ROOT / ".env", override=True)
    provider = os.getenv("EMBEDDING_PROVIDER", "qwen").lower()

    # Fast path when not on Voyage free-tier constraints
    if provider != "voyage":
        return Chroma.from_documents(
            documents=documents,
            embedding=embeddings,
            persist_directory=str(persist_directory),
            collection_name=collection_name,
        )

    batch_size = batch_size or int(os.getenv("VOYAGE_INDEX_BATCH_SIZE", "16"))
    sleep_seconds = (
        sleep_seconds
        if sleep_seconds is not None
        else float(os.getenv("VOYAGE_INDEX_SLEEP_SEC", "21"))
    )

    vectorstore = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=str(persist_directory),
    )

    total = len(documents)
    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        batch = documents[start:end]
        print(f"[chroma] embedding docs {start + 1}-{end}/{total} ...")
        vectorstore.add_documents(batch)
        if end < total and sleep_seconds > 0:
            print(f"[chroma] sleeping {sleep_seconds:.0f}s (Voyage free-tier RPM)...")
            time.sleep(sleep_seconds)

    return vectorstore


def load_chroma(
    embeddings: Embeddings | None = None,
    persist_directory: Path | None = None,
    collection_name: str = COLLECTION_NAME,
) -> Chroma:
    embeddings = embeddings or get_embeddings()
    persist_directory = Path(persist_directory or get_chroma_dir())
    if not persist_directory.exists() or not any(persist_directory.iterdir()):
        raise FileNotFoundError(
            f"Chroma index not found at {persist_directory}. "
            "Run notebook 02 Step 5 (build_chroma) first."
        )
    return Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=str(persist_directory),
    )


def search(query: str, k: int = 5, vectorstore: Chroma | None = None) -> list[Document]:
    vs = vectorstore or load_chroma()
    retriever = vs.as_retriever(search_kwargs={"k": k})
    return list(retriever.invoke(query))


def format_hits(docs: list[Document]) -> str:
    lines: list[str] = []
    for i, doc in enumerate(docs, start=1):
        m = doc.metadata
        lines.append(
            f"{i}. {m.get('name', '?')} — {m.get('tagline', '')}\n"
            f"   votes={m.get('votes_count')} | topics={m.get('topics')} | "
            f"category={m.get('main_category')} | platforms={m.get('platforms')}\n"
            f"   snippet: {doc.page_content[:180]}..."
        )
    return "\n\n".join(lines)
