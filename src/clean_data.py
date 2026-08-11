"""Clean Product Hunt features into searchable_text and saas_cleaned.csv."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = ROOT / "data" / "producthunt_features.csv"
OUT_PATH = ROOT / "data" / "saas_cleaned.csv"

# Drop products whose tagline + description is shorter than this (chars)
MIN_TEXT_CHARS = 40

# Binary source flags → human-readable topic labels for searchable_text
TOPIC_FLAG_MAP = {
    "is_ai_product": "AI",
    "is_saas_product": "SaaS",
    "is_dev_tool": "Developer Tools",
    "is_productivity": "Productivity",
}

# First match wins when a product has multiple topic flags
CATEGORY_PRIORITY = ["SaaS", "AI", "Developer Tools", "Productivity"]

# Columns kept in the cleaned export (business fields + engineered features)
CORE_COLUMNS = [
    "id",
    "name",
    "tagline",
    "description",
    "votes_count",
    "comments_count",
    "topics",
    "main_category",
    "platforms",
    "searchable_text",
    "log_votes",
    "engagement_ratio",
    "maker_count",
    "reviews_count",
    "reviews_rating",
    "media_count",
    "launch_hour_utc",
    "launch_day_of_week",
    "is_weekend",
    "is_optimal_launch_hour",
    "is_optimal_launch_day",
    "name_len_chars",
    "tagline_len_words",
    "desc_len_words",
    "has_emoji_in_tagline",
    "is_question_tagline",
    "topic_count",
    "is_ai_product",
    "is_saas_product",
    "is_dev_tool",
    "is_productivity",
    "is_github_repo",
    "has_custom_domain",
    "daily_rank_clean",
    "weekly_rank_clean",
    "is_viral",
]


def build_topics(row: pd.Series) -> str:
    """Join active topic flags into a comma-separated topic string."""
    labels = [label for col, label in TOPIC_FLAG_MAP.items() if int(row.get(col, 0) or 0) == 1]
    return ", ".join(labels) if labels else "General"


def build_main_category(topics: str) -> str:
    """Pick one primary category using CATEGORY_PRIORITY."""
    present = {t.strip() for t in topics.split(",") if t.strip()}
    for cat in CATEGORY_PRIORITY:
        if cat in present:
            return cat
    return "General"


def build_searchable_text(row: pd.Series) -> str:
    """Compose the single text field used as the embedding input."""
    description = row.get("description")
    if pd.isna(description):
        description = ""
    return (
        f"{row['name']}. {row['tagline']}. {description}. "
        f"Topics: {row['topics']}. Category: {row['main_category']}"
    ).strip()


def missingness_report(df: pd.DataFrame, focus: list[str]) -> pd.DataFrame:
    """Summarize null rates for focus columns (marks absent columns as N/A)."""
    rows = []
    for col in focus:
        if col not in df.columns:
            rows.append(
                {
                    "column": col,
                    "missing": "N/A",
                    "missing_pct": "N/A",
                    "note": "column absent in source",
                }
            )
            continue
        miss = int(df[col].isna().sum())
        rows.append(
            {
                "column": col,
                "missing": miss,
                "missing_pct": round(100 * miss / len(df), 2),
                "note": "ok",
            }
        )
    return pd.DataFrame(rows)


def clean(df: pd.DataFrame, min_text_chars: int = MIN_TEXT_CHARS) -> tuple[pd.DataFrame, dict]:
    """
    Full cleaning pipeline:
    normalize text → synthesize topics/category/searchable_text →
    filter short copy → add ranking features → select export columns.
    """
    work = df.copy()

    for col in ["tagline", "description"]:
        if col in work.columns:
            work[col] = work[col].fillna("").astype(str).str.strip()

    # Source CSV has topic flags, not a free-text topics column
    work["topics"] = work.apply(build_topics, axis=1)
    work["main_category"] = work["topics"].map(build_main_category)
    work["searchable_text"] = work.apply(build_searchable_text, axis=1)

    text_len = (
        work["tagline"].fillna("").astype(str).str.len()
        + work["description"].fillna("").astype(str).str.len()
    )
    before = len(work)
    work = work.loc[text_len >= min_text_chars].copy()
    dropped_short = before - len(work)

    # Ranking helpers for later re-ranking (semantic recall + popularity)
    votes = work["votes_count"].fillna(0).astype(float)
    comments = work["comments_count"].fillna(0).astype(float)
    work["log_votes"] = np.log1p(votes)
    work["engagement_ratio"] = comments / (votes + 1.0)

    keep = [c for c in CORE_COLUMNS if c in work.columns]
    cleaned = work[keep].reset_index(drop=True)

    stats = {
        "raw_rows": before,
        "cleaned_rows": len(cleaned),
        "dropped_short_text": dropped_short,
        "min_text_chars": min_text_chars,
        "main_category_counts": cleaned["main_category"].value_counts().to_dict(),
    }
    return cleaned, stats


def main() -> None:
    if not RAW_PATH.exists():
        raise FileNotFoundError(f"Missing raw data: {RAW_PATH}")

    df = pd.read_csv(RAW_PATH)
    focus_cols = ["tagline", "description", "topics", "name"]
    report = missingness_report(df, focus_cols)
    print("Missingness (focus columns):")
    print(report.to_string(index=False))
    print(f"\nRaw shape: {df.shape}")

    cleaned, stats = clean(df)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(OUT_PATH, index=False)

    print("\nCleaning stats:")
    for k, v in stats.items():
        print(f"  {k}: {v}")
    print(f"\nWrote {OUT_PATH} ({cleaned.shape[0]} rows, {cleaned.shape[1]} cols)")


if __name__ == "__main__":
    main()
