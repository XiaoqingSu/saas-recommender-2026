"""Optional tone / positioning / heat labels for SaaS products."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

import pandas as pd
from dotenv import load_dotenv

from src.classify import (
    CLEANED_PATH,
    ENRICHED_PATH,
    build_classify_text,
    load_zero_shot_pipeline,
    zero_shot_label,
)

ROOT = Path(__file__).resolve().parents[1]
POSITIONED_PATH = ROOT / "data" / "saas_positioned.csv"

TONE_LABELS = ["joy", "professional", "innovative", "practical"]
POSITIONING_LABELS = [
    "Highly Viral",
    "Niche",
    "Enterprise-Ready",
    "Indie Hacker Friendly",
]


def load_base(path: Path | None = None, sample_size: int | None = None) -> pd.DataFrame:
    """Prefer enriched CSV from notebook 03; fall back to cleaned."""
    src = path or (ENRICHED_PATH if ENRICHED_PATH.exists() else CLEANED_PATH)
    df = pd.read_csv(src)
    if sample_size is not None:
        df = df.head(sample_size).copy()
    return df


def heat_tier_from_votes(df: pd.DataFrame) -> pd.Series:
    """
    High Heat vs Long-tail Utility from upvotes / is_viral.
    Threshold: max(median, 75th percentile * 0.6) or is_viral==1.
    """
    votes = df["votes_count"].fillna(0).astype(float)
    p75 = float(votes.quantile(0.75)) if len(votes) else 0.0
    median = float(votes.median()) if len(votes) else 0.0
    threshold = max(median, p75 * 0.6)

    viral = df["is_viral"].fillna(0).astype(int) if "is_viral" in df.columns else 0
    high = (votes >= threshold) | (viral == 1)
    return high.map({True: "High Heat", False: "Long-tail Utility"})


def add_heat_labels(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["heat_tier"] = heat_tier_from_votes(out)
    return out


def add_zero_shot_tone_positioning(
    df: pd.DataFrame,
    *,
    show_progress: bool = True,
) -> pd.DataFrame:
    """Tone + positioning via facebook/bart-large-mnli (same stack as notebook 03)."""
    out = df.copy()
    for col in ["tone", "tone_score", "positioning", "positioning_score"]:
        out[col] = 0.0 if col.endswith("_score") else ""

    classifier = load_zero_shot_pipeline()
    n = len(out)
    for i, (idx, row) in enumerate(out.iterrows(), start=1):
        text = build_classify_text(row)
        if show_progress and (i == 1 or i % 10 == 0 or i == n):
            print(f"[positioning/zero-shot] {i}/{n}: {row.get('name', '')}", flush=True)

        tone, tone_score = zero_shot_label(classifier, text, TONE_LABELS)
        pos, pos_score = zero_shot_label(classifier, text, POSITIONING_LABELS)
        out.at[idx, "tone"] = tone
        out.at[idx, "tone_score"] = round(tone_score, 4)
        out.at[idx, "positioning"] = pos
        out.at[idx, "positioning_score"] = round(pos_score, 4)
    return out


def _deepseek_client():
    load_dotenv(ROOT / ".env", override=True)
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise ValueError("DEEPSEEK_API_KEY missing in .env")
    from openai import OpenAI

    return OpenAI(api_key=api_key, base_url="https://api.deepseek.com")


def llm_positioning_label(client: Any, text: str) -> str:
    """Ask DeepSeek for one positioning label from the closed set."""
    prompt = (
        "Classify this SaaS product into exactly ONE label:\n"
        f"{', '.join(POSITIONING_LABELS)}\n\n"
        f"Product copy:\n{text[:900]}\n\n"
        'Return JSON only: {"positioning": "<label>"}'
    )
    resp = client.chat.completions.create(
        model=os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
        messages=[
            {"role": "system", "content": "You label SaaS products. Reply with JSON only."},
            {"role": "user", "content": prompt},
        ],
        temperature=0,
    )
    content = resp.choices[0].message.content or ""
    match = re.search(r"\{.*\}", content, flags=re.S)
    if match:
        data = json.loads(match.group(0))
        label = str(data.get("positioning", "")).strip()
        if label in POSITIONING_LABELS:
            return label
    for label in POSITIONING_LABELS:
        if label.lower() in content.lower():
            return label
    return "Niche"


def add_llm_positioning(
    df: pd.DataFrame,
    *,
    show_progress: bool = True,
) -> pd.DataFrame:
    """Optional DeepSeek positioning labels (overrides zero-shot positioning if present)."""
    out = df.copy()
    if "positioning" not in out.columns:
        out["positioning"] = ""
    out["positioning_source"] = "llm"

    client = _deepseek_client()
    n = len(out)
    for i, (idx, row) in enumerate(out.iterrows(), start=1):
        text = build_classify_text(row)
        if show_progress and (i == 1 or i % 10 == 0 or i == n):
            print(f"[positioning/llm] {i}/{n}: {row.get('name', '')}", flush=True)
        out.at[idx, "positioning"] = llm_positioning_label(client, text)
    return out


def enrich_positioning(
    df: pd.DataFrame,
    *,
    use_zero_shot: bool = True,
    use_llm_positioning: bool = False,
    show_progress: bool = True,
) -> pd.DataFrame:
    """
    Pipeline:
      1) heat_tier from upvotes
      2) optional zero-shot tone + positioning
      3) optional LLM positioning override
    """
    out = add_heat_labels(df)
    if use_zero_shot:
        out = add_zero_shot_tone_positioning(out, show_progress=show_progress)
        out["positioning_source"] = "zero-shot"
    else:
        out["tone"] = ""
        out["tone_score"] = 0.0
        out["positioning"] = ""
        out["positioning_score"] = 0.0
        out["positioning_source"] = "none"

    if use_llm_positioning:
        out = add_llm_positioning(out, show_progress=show_progress)
    return out


def filter_positioning(
    df: pd.DataFrame,
    *,
    heat_tier: str | None = None,
    tone: str | None = None,
    positioning: str | None = None,
) -> pd.DataFrame:
    out = df
    if heat_tier and heat_tier != "All":
        out = out[out["heat_tier"] == heat_tier]
    if tone and tone != "All":
        out = out[out["tone"] == tone]
    if positioning and positioning != "All":
        out = out[out["positioning"] == positioning]
    return out.reset_index(drop=True)


def main() -> None:
    sample = os.getenv("POSITION_SAMPLE_SIZE")
    sample_size = int(sample) if sample else None
    use_llm = os.getenv("USE_LLM_POSITIONING", "0") == "1"
    df = load_base(sample_size=sample_size)
    print(f"Loaded {len(df)} rows")
    out = enrich_positioning(df, use_zero_shot=True, use_llm_positioning=use_llm)
    POSITIONED_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(POSITIONED_PATH, index=False)
    print(f"Wrote {POSITIONED_PATH} ({out.shape[0]} rows, {out.shape[1]} cols)")
    print("\nheat_tier:\n", out["heat_tier"].value_counts().to_string())
    if "tone" in out.columns:
        print("\ntone:\n", out["tone"].value_counts().to_string())
    if "positioning" in out.columns:
        print("\npositioning:\n", out["positioning"].value_counts().to_string())


if __name__ == "__main__":
    main()
