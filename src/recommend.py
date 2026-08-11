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


def _esc(text: object) -> str:
    return (
        str(text or "")
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def format_results_markdown(results: list[dict], query: str, filters: dict) -> str:
    """HTML result cards for the Gradio frontend."""
    if not query.strip():
        return (
            "<p class='hint'>Describe what you need — e.g. "
            "<em>AI coding agent for solo founders</em>.</p>"
        )
    if not results:
        return (
            "<div class='empty'>"
            "<h3>No matches</h3>"
            f"<p>Query: <strong>{_esc(query)}</strong></p>"
            f"<p>Filters: {_esc(filters)}</p>"
            "<p>Widen category / heat / audience, or rebuild a larger index.</p>"
            "</div>"
        )

    cards: list[str] = [
        "<div class='results-head'>",
        f"<h2>Results for “{_esc(query.strip())}”</h2>",
        (
            f"<p class='meta'>category · {_esc(filters.get('category'))} &nbsp;|&nbsp; "
            f"heat · {_esc(filters.get('heat'))} &nbsp;|&nbsp; "
            f"audience · {_esc(filters.get('audience'))}</p>"
        ),
        "</div>",
        "<div class='result-list'>",
    ]

    for i, r in enumerate(results, start=1):
        badges = []
        if r.get("product_type"):
            badges.append(f"<span class='badge'>{_esc(r['product_type'])}</span>")
        if r.get("heat_tier"):
            badges.append(f"<span class='badge badge-heat'>{_esc(r['heat_tier'])}</span>")
        if r.get("target_user"):
            badges.append(f"<span class='badge badge-user'>{_esc(r['target_user'])}</span>")
        badge_html = " ".join(badges)

        cards.append(
            f"""
<article class="result-card">
  <div class="rank">#{i}</div>
  <div class="body">
    <h3>{_esc(r['name'])}</h3>
    <p class="tagline">{_esc(r['tagline'])}</p>
    <div class="badges">{badge_html}</div>
    <p class="votes"><strong>{_esc(r['votes_count'])}</strong> upvotes
       · topics: {_esc(r.get('topics') or '—')}</p>
    <p class="links">
      <span>{_esc(r.get('website') or 'N/A')}</span>
      · <a href="{_esc(r['product_hunt_url'])}" target="_blank" rel="noopener">Product Hunt</a>
    </p>
    <p class="why"><strong>Why:</strong> {_esc(r.get('reason'))}</p>
  </div>
</article>
"""
        )

    cards.append("</div>")
    return "\n".join(cards)
