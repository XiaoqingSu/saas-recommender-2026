"""Gradio dashboard: NL SaaS search + category / heat / audience filters."""

from __future__ import annotations

import gradio as gr

from src.recommend import (
    HEAT_OPTIONS,
    format_results_markdown,
    recommend,
    resolve_catalog_path,
)
from src.vector_search import get_chroma_dir

CATEGORIES = ["All", "SaaS", "AI", "Developer Tools", "Productivity", "General"]
AUDIENCES = ["All", "Solo Founder", "SMB", "Enterprise"]


def run_recommend(query: str, category: str, heat: str, audience: str) -> str:
    filters = {"category": category, "heat": heat, "audience": audience}
    try:
        results = recommend(
            query,
            category=category,
            heat=heat,
            audience=audience,
            top_k=5,
            recall_k=30,
        )
    except FileNotFoundError as exc:
        return (
            f"**Index missing.** {exc}\n\n"
            "Run `notebooks/02_vector_search.ipynb` Steps 0–5 first."
        )
    except Exception as exc:  # noqa: BLE001 — surface errors in UI
        return f"**Error:** `{type(exc).__name__}: {exc}`"
    return format_results_markdown(results, query, filters)


def build_demo() -> gr.Blocks:
    catalog = resolve_catalog_path().name
    chroma = get_chroma_dir().name

    with gr.Blocks(title="SaaS Recommender 2026") as demo:
        gr.Markdown(
            f"""
# SaaS Recommender
Natural-language SaaS discovery on 2026 Product Hunt launches.

<small>Catalog: <code>{catalog}</code> · Vector index: <code>{chroma}</code>  
First query may take 30–90s while the embedding model loads.</small>
"""
        )

        query = gr.Textbox(
            label="What do you need?",
            placeholder="I need a tool that auto-generates weekly reports and integrates with Slack",
            lines=3,
        )

        with gr.Row():
            category = gr.Dropdown(
                CATEGORIES,
                value="All",
                label="Category",
                info="SaaS / AI / Developer Tools / Productivity",
            )
            heat = gr.Dropdown(
                HEAT_OPTIONS,
                value="All",
                label="Heat filter",
                info="Upvote thresholds or High Heat only",
            )
            audience = gr.Dropdown(
                AUDIENCES,
                value="All",
                label="Target user",
                info="From notebook 03 classification (if available)",
            )

        submit = gr.Button("Recommend", variant="primary")
        output = gr.Markdown()

        gr.Examples(
            examples=[
                ["AI coding agent for solo founders", "AI", "All", "Solo Founder"],
                [
                    "low-cost freemium SaaS for collecting customer feedback",
                    "SaaS",
                    "Votes ≥ 50",
                    "All",
                ],
                [
                    "privacy-first AI browser for Mac",
                    "All",
                    "High Heat only",
                    "All",
                ],
            ],
            inputs=[query, category, heat, audience],
        )

        submit.click(run_recommend, [query, category, heat, audience], output)
        query.submit(run_recommend, [query, category, heat, audience], output)

    return demo


if __name__ == "__main__":
    build_demo().launch()
