"""Enrich SaaS products with product_type + zero-shot filter labels."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pandas as pd
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
CLEANED_PATH = ROOT / "data" / "saas_cleaned.csv"
ENRICHED_PATH = ROOT / "data" / "saas_enriched.csv"

ZERO_SHOT_MODEL = os.getenv("ZERO_SHOT_MODEL", "facebook/bart-large-mnli")

LABEL_SETS: dict[str, list[str]] = {
    "target_user": ["Solo Founder", "SMB", "Enterprise"],
    "pricing_model": ["Free", "Freemium", "Paid"],
    "core_function": [
        "Analytics",
        "Automation",
        "Collaboration",
        "AI Agent",
        "Developer Tool",
    ],
}


def load_cleaned(path: Path = CLEANED_PATH, sample_size: int | None = None) -> pd.DataFrame:
    df = pd.read_csv(path)
    if sample_size is not None:
        df = df.head(sample_size).copy()
    return df


def product_type_from_flags(row: pd.Series) -> str:
    """Map existing engineered flags / topics to a product_type facet."""
    if int(row.get("is_saas_product", 0) or 0) == 1:
        return "SaaS"
    if int(row.get("is_ai_product", 0) or 0) == 1:
        return "AI"
    if int(row.get("is_dev_tool", 0) or 0) == 1:
        return "Developer Tools"
    if int(row.get("is_productivity", 0) or 0) == 1:
        return "Productivity"
    main = str(row.get("main_category", "") or "").strip()
    return main if main else "General"


def build_classify_text(row: pd.Series, max_chars: int = 800) -> str:
    """Short text fed to zero-shot (keeps inference cheaper/faster)."""
    parts = [
        str(row.get("name", "") or ""),
        str(row.get("tagline", "") or ""),
        str(row.get("description", "") or ""),
        f"Topics: {row.get('topics', '')}",
    ]
    text = ". ".join(p.strip() for p in parts if p and str(p).strip())
    return text[:max_chars]


def load_zero_shot_pipeline(model_name: str = ZERO_SHOT_MODEL):
    load_dotenv(ROOT / ".env", override=True)
    hf_token = os.getenv("HF_TOKEN")
    if hf_token:
        os.environ["HUGGING_FACE_HUB_TOKEN"] = hf_token
        os.environ["HF_TOKEN"] = hf_token

    from transformers import pipeline

    device = 0 if os.getenv("ZERO_SHOT_DEVICE", "cpu").lower() == "cuda" else -1
    return pipeline(
        "zero-shot-classification",
        model=model_name,
        device=device,
    )


# Back-compat alias
_load_zero_shot_pipeline = load_zero_shot_pipeline


def zero_shot_label(
    classifier: Any,
    text: str,
    candidate_labels: list[str],
    multi_label: bool = False,
) -> tuple[str, float]:
    result = classifier(text, candidate_labels, multi_label=multi_label)
    labels = result["labels"]
    scores = result["scores"]
    return str(labels[0]), float(scores[0])


def enrich(
    df: pd.DataFrame,
    *,
    use_zero_shot: bool = True,
    model_name: str = ZERO_SHOT_MODEL,
    show_progress: bool = True,
) -> pd.DataFrame:
    """
    Add product_type from flags, then optional zero-shot labels:
    target_user / pricing_model / core_function (+ confidence columns).
    """
    out = df.copy()
    out["product_type"] = out.apply(product_type_from_flags, axis=1)

    for col in [
        "target_user",
        "target_user_score",
        "pricing_model",
        "pricing_model_score",
        "core_function",
        "core_function_score",
    ]:
        if col.endswith("_score"):
            out[col] = 0.0
        else:
            out[col] = ""

    if not use_zero_shot:
        return out

    classifier = _load_zero_shot_pipeline(model_name)
    n = len(out)
    for i, (idx, row) in enumerate(out.iterrows(), start=1):
        text = build_classify_text(row)
        if show_progress and (i == 1 or i % 10 == 0 or i == n):
            print(f"[classify] {i}/{n}: {row.get('name', '')}", flush=True)

        for field, labels in LABEL_SETS.items():
            label, score = zero_shot_label(classifier, text, labels, multi_label=False)
            out.at[idx, field] = label
            out.at[idx, f"{field}_score"] = round(score, 4)

    return out


def filter_enriched(
    df: pd.DataFrame,
    *,
    product_type: str | None = None,
    target_user: str | None = None,
    pricing_model: str | None = None,
    core_function: str | None = None,
) -> pd.DataFrame:
    """Apply facet filters used by the dashboard after semantic recall."""
    out = df
    if product_type and product_type != "All":
        out = out[out["product_type"] == product_type]
    if target_user and target_user != "All":
        out = out[out["target_user"] == target_user]
    if pricing_model and pricing_model != "All":
        out = out[out["pricing_model"] == pricing_model]
    if core_function and core_function != "All":
        out = out[out["core_function"] == core_function]
    return out.reset_index(drop=True)


def main() -> None:
    sample = os.getenv("CLASSIFY_SAMPLE_SIZE")
    sample_size = int(sample) if sample else None
    df = load_cleaned(sample_size=sample_size)
    print(f"Loaded {len(df)} rows from {CLEANED_PATH}")
    enriched = enrich(df, use_zero_shot=True)
    ENRICHED_PATH.parent.mkdir(parents=True, exist_ok=True)
    enriched.to_csv(ENRICHED_PATH, index=False)
    print(f"Wrote {ENRICHED_PATH} ({enriched.shape[0]} rows, {enriched.shape[1]} cols)")
    print("\nproduct_type:\n", enriched["product_type"].value_counts().to_string())
    print("\ntarget_user:\n", enriched["target_user"].value_counts().to_string())
    print("\npricing_model:\n", enriched["pricing_model"].value_counts().to_string())
    print("\ncore_function:\n", enriched["core_function"].value_counts().to_string())


if __name__ == "__main__":
    main()
