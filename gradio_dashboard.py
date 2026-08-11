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

CSS = """
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;700&family=Fraunces:opsz,wght@9..144,600;9..144,700&display=swap');

:root {
  --bg0: #f3f0e8;
  --bg1: #e7eef2;
  --ink: #1c2430;
  --muted: #5c6b7a;
  --accent: #0f6b5c;
  --accent-soft: #d7efe9;
  --card: rgba(255,255,255,0.82);
  --line: #d5ddd8;
}

.gradio-container {
  font-family: "DM Sans", sans-serif !important;
  color: var(--ink) !important;
  max-width: 980px !important;
}

body, .gradio-container {
  background:
    radial-gradient(1200px 500px at 10% -10%, #dceee8 0%, transparent 55%),
    radial-gradient(900px 420px at 100% 0%, #e8e2d4 0%, transparent 50%),
    linear-gradient(180deg, var(--bg0), var(--bg1)) !important;
}

#brand h1 {
  font-family: "Fraunces", Georgia, serif !important;
  font-size: 2.4rem !important;
  letter-spacing: -0.02em;
  margin-bottom: 0.2rem !important;
  color: var(--ink) !important;
}

#brand p {
  color: var(--muted) !important;
  font-size: 1.02rem !important;
}

#brand .meta {
  color: var(--muted);
  font-size: 0.86rem;
}

button.primary {
  background: var(--accent) !important;
  border: none !important;
}

.results-head h2 {
  font-family: "Fraunces", Georgia, serif;
  margin: 0.2rem 0 0.35rem;
}

.results-head .meta, .hint, .empty p {
  color: var(--muted);
}

.result-list {
  display: grid;
  gap: 0.9rem;
  margin-top: 0.8rem;
}

.result-card {
  display: grid;
  grid-template-columns: 48px 1fr;
  gap: 0.75rem;
  background: var(--card);
  border: 1px solid var(--line);
  border-radius: 16px;
  padding: 1rem 1.1rem;
  backdrop-filter: blur(6px);
}

.result-card .rank {
  width: 40px;
  height: 40px;
  border-radius: 12px;
  background: var(--accent-soft);
  color: var(--accent);
  display: flex;
  align-items: center;
  justify-content: center;
  font-weight: 700;
}

.result-card h3 {
  margin: 0;
  font-family: "Fraunces", Georgia, serif;
  font-size: 1.25rem;
}

.result-card .tagline {
  margin: 0.25rem 0 0.55rem;
  color: var(--muted);
}

.badges { display: flex; flex-wrap: wrap; gap: 0.35rem; margin-bottom: 0.45rem; }
.badge {
  font-size: 0.75rem;
  padding: 0.18rem 0.5rem;
  border-radius: 999px;
  background: #eef3f1;
  color: #314349;
  border: 1px solid #d7e1dc;
}
.badge-heat { background: #fff1df; border-color: #f0d2a8; color: #8a4b12; }
.badge-user { background: #e7eefc; border-color: #c8d7f5; color: #27457a; }

.result-card .votes, .result-card .links, .result-card .why {
  margin: 0.25rem 0;
  font-size: 0.92rem;
}
.result-card a { color: var(--accent); }
.result-card .why { color: #3d4a55; }
"""


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
            f"<div class='empty'><h3>Index missing</h3><p>{exc}</p>"
            "<p>Run notebooks/02_vector_search.ipynb Steps 0–5 first.</p></div>"
        )
    except Exception as exc:  # noqa: BLE001 — surface errors in UI
        return f"<div class='empty'><h3>Error</h3><p><code>{type(exc).__name__}: {exc}</code></p></div>"
    return format_results_markdown(results, query, filters)


def build_demo() -> gr.Blocks:
    catalog = resolve_catalog_path().name
    chroma = get_chroma_dir().name

    theme = gr.themes.Soft(
        primary_hue="teal",
        secondary_hue="stone",
        neutral_hue="stone",
        font=gr.themes.GoogleFont("DM Sans"),
    )

    with gr.Blocks(title="SaaS Recommender 2026", theme=theme, css=CSS) as demo:
        with gr.Column(elem_id="brand"):
            gr.Markdown(
                f"""
# SaaS Recommender
Natural-language discovery across 2026 Product Hunt launches.

<p class="meta">Catalog: <code>{catalog}</code> · Index: <code>{chroma}</code><br/>
First query may take 30–90s while the embedding model loads.</p>
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
            )
            heat = gr.Dropdown(
                HEAT_OPTIONS,
                value="All",
                label="Heat filter",
            )
            audience = gr.Dropdown(
                AUDIENCES,
                value="All",
                label="Target user",
            )

        submit = gr.Button("Recommend", variant="primary")
        output = gr.HTML()

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
    build_demo().launch(server_name="127.0.0.1", server_port=7860, inbrowser=True)
