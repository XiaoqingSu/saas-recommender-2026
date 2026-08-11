# SaaS Recommender (2026)

Natural-language SaaS discovery over **2026 Product Hunt** launches.

**Repo:** [github.com/XiaoqingSu/saas-recommender-2026](https://github.com/XiaoqingSu/saas-recommender-2026)

---

## Why this project

In 2026, new SaaS and AI tools launch every week—this recommender turns that noise into actionable picks via natural-language search over live Product Hunt data, with filters for category, heat, and target audience.

---

## Architecture

```text
data/producthunt_features.csv          # raw 31 engineered columns
            │
            ▼
01 Data cleaning                       # searchable_text + saas_cleaned.csv
            │
            ▼
02 Vector search                       # Qwen3 embeddings → Chroma Top-K recall
            │
            ▼
03 Classification                      # product_type + zero-shot facets
            │
            ▼
04 Sentiment / positioning (optional)  # tone + positioning + heat_tier
            │
            ▼
Gradio dashboard                       # NL query + filters + result cards
```

### Retrieval flow (runtime)

1. User enters a natural-language need.
2. Query is embedded with **Qwen/Qwen3-Embedding-0.6B** (local).
3. Chroma returns Top-K semantic candidates (default recall `k=30`).
4. Candidates are joined to the catalog (`saas_positioned.csv` → enriched → cleaned).
5. Facet filters apply: category / heat / audience.
6. Survivors are re-ranked by `votes_count` and shown in the UI.

---

## Core fields

### Embedding input

```text
searchable_text = "{name}. {tagline}. {description}. Topics: {topics}. Category: {main_category}"
```

### Ranking helpers (from cleaning)

- `votes_count`, `log_votes`, `engagement_ratio`, `is_viral`
- Launch timing / topic flags retained for later weighting

### Filter facets

| Facet | Source | Values |
| --- | --- | --- |
| Category / `product_type` | Topic flags | SaaS, AI, Developer Tools, Productivity, General |
| Target user | Zero-shot (`facebook/bart-large-mnli`) | Solo Founder, SMB, Enterprise |
| Pricing model | Zero-shot | Free, Freemium, Paid |
| Core function | Zero-shot | Analytics, Automation, Collaboration, AI Agent, Developer Tool |
| Heat tier | Upvotes / `is_viral` | High Heat, Long-tail Utility |
| Tone | Zero-shot | joy, professional, innovative, practical |
| Positioning | Zero-shot (optional DeepSeek) | Highly Viral, Niche, Enterprise-Ready, Indie Hacker Friendly |

---

## Project structure

```text
saas-recommender/
├── data/
│   ├── producthunt_features.csv   # raw ~5.6k rows × 31 engineered columns
│   ├── saas_cleaned.csv           # searchable_text + ranking features
│   ├── saas_enriched.csv          # + product_type / audience / pricing / function
│   └── saas_positioned.csv        # + tone / positioning / heat_tier
├── notebooks/
│   ├── 01_data_cleaning.ipynb
│   ├── 02_vector_search.ipynb
│   ├── 03_classification.ipynb
│   └── 04_sentiment_or_positioning.ipynb
├── src/
│   ├── clean_data.py              # cleaning pipeline
│   ├── vector_search.py           # embeddings + Chroma
│   ├── classify.py                # zero-shot enrichment
│   ├── positioning.py             # tone / positioning / heat
│   └── recommend.py               # recall + filter + UI formatting
├── gradio_dashboard.py            # frontend
├── .env.example                   # env template (no secrets)
├── pyproject.toml                 # UV project / dependencies
├── uv.lock
├── requirements.txt               # pip-compatible export
└── README.md
```

Local artifacts (not committed):

- `.env` — API tokens / local settings
- `.venv/` — virtualenv
- `.chroma/` — persisted vector index (model-specific folder names)

---

## Quick start (UV)

Requires **Python 3.12+** and [uv](https://github.com/astral-sh/uv).

```bash
# 1) Install dependencies
uv sync

# 2) Create env file
copy .env.example .env   # Windows
# cp .env.example .env   # macOS / Linux

# 3) (Optional) set HF_TOKEN in .env for faster model downloads

# 4) Clean data
uv run python -m src.clean_data

# 5) Build / refresh vector index (see notebook 02 for guided steps)
#    Or run notebooks/02_vector_search.ipynb Steps 0–5

# 6) Launch the Gradio UI
uv run python gradio_dashboard.py
```

Open **http://127.0.0.1:7860**

Alternative install:

```bash
pip install -r requirements.txt
```

---

## Environment variables

See `.env.example`. Important keys:

| Variable | Default | Purpose |
| --- | --- | --- |
| `EMBEDDING_PROVIDER` | `qwen` | `qwen` \| `voyage` \| `openai` \| `local` |
| `QWEN_EMBEDDING_MODEL` | `Qwen/Qwen3-Embedding-0.6B` | Local embedding model (1024-d) |
| `HF_TOKEN` | empty | Hugging Face Hub auth / higher rate limits |
| `ZERO_SHOT_MODEL` | `facebook/bart-large-mnli` | Classification + tone/positioning |
| `ZERO_SHOT_DEVICE` | `cpu` | Use `cuda` if you have a GPU |
| `CLASSIFY_SAMPLE_SIZE` | `40` | CLI sample size for notebook 03 pipeline |
| `POSITION_SAMPLE_SIZE` | `40` | CLI sample size for notebook 04 pipeline |
| `DEEPSEEK_API_KEY` | empty | Optional LLM positioning labels |
| `DEEPSEEK_MODEL` | `deepseek-chat` | DeepSeek chat model id |
| `USE_LLM_POSITIONING` | `0` | `1` to override positioning via DeepSeek |

**Never commit `.env`.** Only `.env.example` is tracked.

---

## Notebooks (step-by-step)

### 01 — Data cleaning

- Load `producthunt_features.csv`
- Check missing values (`tagline`, `description`)
- Synthesize `topics` / `main_category` from binary flags (source has no free-text topics)
- Build `searchable_text`, drop ultra-short copy, add `log_votes` / `engagement_ratio`
- Export `data/saas_cleaned.csv`

```bash
uv run python -m src.clean_data
```

### 02 — Vector search

- Split `searchable_text` with LangChain `CharacterTextSplitter`
- Embed with **Qwen3-Embedding-0.6B** (default)
- Persist Chroma under `.chroma/saas_products_qwen_...`
- Metadata keeps name, tagline, votes, topics, platforms, engineered flags
- Smoke-test NL queries

Notes:

- First local model download can take a few minutes.
- `SAMPLE_SIZE=500` is recommended while learning; set `None` for the full ~5.6k catalog.
- Changing embedding model requires rebuilding the Chroma index (dimensions differ).

### 03 — Classification

- Map flags → `product_type`
- Zero-shot labels for `target_user`, `pricing_model`, `core_function`
- Export `data/saas_enriched.csv`

```bash
# PowerShell example
$env:CLASSIFY_SAMPLE_SIZE=40
uv run python -m src.classify
```

CPU zero-shot is slow; start with a sample, then scale up.

### 04 — Sentiment / positioning (optional)

- `heat_tier` from upvotes / `is_viral`
- Zero-shot `tone` + `positioning`
- Optional DeepSeek override (`USE_LLM_POSITIONING=1`)
- Export `data/saas_positioned.csv`

```bash
$env:POSITION_SAMPLE_SIZE=40
uv run python -m src.positioning
```

---

## Gradio dashboard

```bash
uv run python gradio_dashboard.py
```

UI controls:

| Control | Behavior |
| --- | --- |
| Text box | Natural-language need |
| Category | SaaS / AI / Developer Tools / Productivity |
| Heat filter | High Heat, Long-tail, or Votes ≥ 50/100/200 |
| Target user | Solo Founder / SMB / Enterprise |
| Results | Name, tagline, upvotes, platforms, Product Hunt search link, short match reason |

The dashboard prefers catalog files in this order:

1. `saas_positioned.csv`
2. `saas_enriched.csv`
3. `saas_cleaned.csv`

First query may take 30–90s while the embedding model loads into memory.

---

## Data notes

- About **5,624** products and **31** engineered columns (votes, launch timing, topic flags, ranks, viral, etc.).
- Source CSV does **not** include free-text `topics`, `website_url`, or `product_hunt_url`.
- Cleaning synthesizes `topics` / `main_category` from flags such as `is_ai_product`, `is_saas_product`, `is_dev_tool`, `is_productivity`.
- The UI shows platform hints and a Product Hunt **search** link as a practical fallback when canonical URLs are unavailable.

---

## Tech stack

- **Package manager:** UV (`pyproject.toml` + `uv.lock`)
- **Orchestration / RAG glue:** LangChain + LangChain-Chroma
- **Embeddings:** Qwen3-Embedding-0.6B via `langchain-huggingface` / sentence-transformers
- **Vector DB:** Chroma (local persistence)
- **Zero-shot NLP:** Hugging Face `facebook/bart-large-mnli`
- **Optional LLM:** DeepSeek (OpenAI-compatible client)
- **UI:** Gradio
- **Data:** pandas

---

## Suggested demo path

1. `uv sync` + copy `.env.example` → `.env`
2. Run cleaning (`01` / `python -m src.clean_data`)
3. Build a 500-row Chroma index (`02`)
4. Enrich a sample with classification + positioning (`03`, `04`)
5. Launch Gradio and try the built-in example queries
6. Scale `SAMPLE_SIZE` to full catalog when ready

---

## License / data

Dataset usage should respect the original Product Hunt / upstream data license and terms. This repo is provided for learning and portfolio demonstration.
