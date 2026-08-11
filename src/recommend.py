"""Recommend SaaS products: semantic recall + facet filters + ranking."""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path
from urllib.parse import quote_plus

import pandas as pd
from langchain_chroma import Chroma

from src.classify import CLEANED_PATH, ENRICHED_PATH
from src.positioning import POSITIONED_PATH
from src.vector_search import get_chroma_dir, get_embeddings, load_chroma, search

ROOT = Path(__file__).resolve().parents[1]

HEAT_OPTIONS = [
    "All",
    "High Heat only",
    "Long-tail Utility only",
    "Votes ≥ 50",
    "Votes ≥ 100",
    "Votes ≥ 200",
]


def resolve_catalog_path() -> Path:
    if POSITIONED_PATH.exists():
        return POSITIONED_PATH
    if ENRICHED_PATH.exists():
        return ENRICHED_PATH
    return CLEANED_PATH


@lru_cache(maxsize=1)
def load_catalog() -> pd.DataFrame:
    path = resolve_catalog_path()
    df = pd.read_csv(path)
    if "product_type" not in df.columns and "main_category" in df.columns:
        df["product_type"] = df["main_category"]
    if "id" in df.columns:
        df["id"] = df["id"].astype(str)
    return df


@lru_cache(maxsize=1)
def get_vectorstore() -> Chroma:
    return load_chroma(embeddings=get_embeddings(), persist_directory=get_chroma_dir())


def slugify(name: str) -> str:
    s = name.lower().strip()
    s = re.sub(r"[^a-z0-9]+", "-", s)
    return s.strip("-")


def product_hunt_url(name: str, product_id: str | None = None) -> str:
    # Source CSV has no canonical PH URL; search link is the reliable fallback.
    return f"https://www.producthunt.com/search?q={quote_plus(name)}"


def website_hint(platforms: str) -> str:
    platforms = str(platforms or "")
    if "Website" in platforms:
        return f"Listed on: {platforms}"
    if platforms.strip():
        return platforms
    return "N/A"


def match_reason(row: pd.Series, query: str) -> str:
    bits = [f'Semantic match for “{query.strip()[:60]}"']
    if row.get("topics"):
        bits.append(f"topics={row.get('topics')}")
    if row.get("product_type"):
        bits.append(f"type={row.get('product_type')}")
    if row.get("target_user"):
        bits.append(f"audience={row.get('target_user')}")
    if row.get("heat_tier"):
        bits.append(f"heat={row.get('heat_tier')}")
    if row.get("core_function"):
        bits.append(f"function={row.get('core_function')}")
    return "; ".join(bits)


def apply_filters(
    df: pd.DataFrame,
    *,
    category: str = "All",
    heat: str = "All",
    audience: str = "All",
) -> pd.DataFrame:
    out = df
    if category and category != "All":
        col = "product_type" if "product_type" in out.columns else "main_category"
        out = out[out[col].astype(str) == category]

    if audience and audience != "All" and "target_user" in out.columns:
        out = out[out["target_user"].astype(str) == audience]

    if heat and heat != "All":
        if heat == "High Heat only" and "heat_tier" in out.columns:
            out = out[out["heat_tier"] == "High Heat"]
        elif heat == "Long-tail Utility only" and "heat_tier" in out.columns:
            out = out[out["heat_tier"] == "Long-tail Utility"]
        elif heat.startswith("Votes ≥"):
            try:
                min_votes = int(heat.split("≥")[-1].strip())
                out = out[out["votes_count"].fillna(0).astype(int) >= min_votes]
            except ValueError:
                pass
    return out


def recommend(
    query: str,
    *,
    category: str = "All",
    heat: str = "All",
    audience: str = "All",
    top_k: int = 5,
    recall_k: int = 30,
) -> list[dict]:
    """
    1) semantic recall from Chroma
    2) join catalog attributes
    3) apply category / heat / audience filters
    4) rank by votes within survivors (stable, useful for heat UX)
    """
    q = (query or "").strip()
    if not q:
        return []

    catalog = load_catalog()
    vs = get_vectorstore()
    hits = search(q, k=recall_k, vectorstore=vs)

    rows: list[dict] = []
    seen: set[str] = set()
    for rank, doc in enumerate(hits, start=1):
        meta = doc.metadata or {}
        pid = str(meta.get("id", ""))
        name = str(meta.get("name", ""))
        key = pid or name
        if not key or key in seen:
            continue
        seen.add(key)

        if pid and "id" in catalog.columns:
            match = catalog[catalog["id"].astype(str) == pid]
        else:
            match = catalog[catalog["name"].astype(str) == name]

        if match.empty:
            # Index may cover products not present in the (smaller) enriched sample.
            row = pd.Series(meta)
        else:
            row = match.iloc[0]

        rows.append(
            {
                "id": pid or str(row.get("id", "")),
                "name": str(row.get("name", name)),
                "tagline": str(row.get("tagline", meta.get("tagline", ""))),
                "votes_count": int(row.get("votes_count", meta.get("votes_count", 0)) or 0),
                "product_type": str(row.get("product_type", row.get("main_category", ""))),
                "topics": str(row.get("topics", meta.get("topics", ""))),
                "target_user": str(row.get("target_user", "")),
                "heat_tier": str(row.get("heat_tier", "")),
                "platforms": str(row.get("platforms", meta.get("platforms", ""))),
                "product_hunt_url": product_hunt_url(str(row.get("name", name)), pid),
                "website": website_hint(str(row.get("platforms", meta.get("platforms", "")))),
                "reason": match_reason(row, q),
                "recall_rank": rank,
            }
        )

    if not rows:
        return []

    frame = pd.DataFrame(rows)
    filtered = apply_filters(frame, category=category, heat=heat, audience=audience)
    if filtered.empty:
        return []

    # Prefer higher upvotes among semantic survivors
    filtered = filtered.sort_values(
        ["votes_count", "recall_rank"],
        ascending=[False, True],
    ).head(top_k)
    return filtered.to_dict(orient="records")


def format_results_markdown(results: list[dict], query: str, filters: dict) -> str:
    if not query.strip():
        return "Enter a natural-language need, e.g. *AI coding agent for solo founders*."
    if not results:
        return (
            "No products matched after filters.\n\n"
            f"Query: **{query}**  \n"
            f"Filters: {filters}\n\n"
            "Try widening category / heat / audience, or rebuild a larger Chroma index."
        )

    lines = [
        f"### Results for “{query.strip()}”",
        f"Filters: category=`{filters.get('category')}` · heat=`{filters.get('heat')}` · audience=`{filters.get('audience')}`",
        f"Catalog: `{resolve_catalog_path().name}` · index: `{get_chroma_dir().name}`",
        "",
    ]
    for i, r in enumerate(results, start=1):
        lines.extend(
            [
                f"#### {i}. {r['name']}",
                f"**{r['tagline']}**",
                f"- Upvotes: **{r['votes_count']}**"
                + (f" · Heat: {r['heat_tier']}" if r.get("heat_tier") else "")
                + (f" · Audience: {r['target_user']}" if r.get("target_user") else "")
                + (f" · Type: {r['product_type']}" if r.get("product_type") else ""),
                f"- Topics: {r.get('topics') or '—'}",
                f"- Website / platforms: {r.get('website') or 'N/A'}",
                f"- Product Hunt: [{r['name']} on PH]({r['product_hunt_url']})",
                f"- Why: {r.get('reason')}",
                "",
            ]
        )
    return "\n".join(lines)
