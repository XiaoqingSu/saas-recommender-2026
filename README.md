# SaaS Recommender (2026)

基于 **2026 Product Hunt** 产品发布数据的自然语言 SaaS 推荐系统。  
从 Book Recommender 架构迁移：用 `name + tagline + description + topics` 替代书摘，用品类 / 热度 / 目标用户替代文学情绪标签。

## 商业价值

| 维度 | Book Recommender | SaaS Recommender |
| --- | --- | --- |
| 文本来源 | 书籍 description | Name + tagline + description + topics |
| 过滤维度 | Fiction / Emotion | 产品品类 + 热度 + 目标用户 |
| 排序信号 | 纯语义相似度 | 语义 + upvote / engagement 加权 |
| 商业用途 | 阅读兴趣 | 创业选型、竞品发现、销售线索 |
| 数据时效 | 历史书籍 | 2026 新产品，叙事更新 |

适合作为 Upwork / 作品集案例：**真实 2026 PH 数据 + 语义推荐 + 多维过滤 + 可演示 Dashboard**。

## 架构

```text
producthunt_features.csv
        ↓
数据清洗与文本标准化（01）
        ↓
向量检索 + Chroma（02，语义召回）
        ↓
分类 / 零样本增强（03）
        ↓
情感 / 定位标签（04，可选）
        ↓
Gradio 交互界面（自然语言查询 + 多维过滤）
```

### 核心字段

- **Embedding 输入**: `searchable_text = "{name}. {tagline}. {description}. Topics: {topics}. Category: {main_category}"`
- **排序加权**: `votes_count` / `log_votes` / `engagement_ratio` / `is_viral`
- **过滤**: `main_category`（SaaS / AI / Developer Tools / Productivity）及后续用户画像标签

## 项目结构

```text
Saas-recommender/
├── data/
│   ├── producthunt_features.csv   # 原始 31 列工程化数据
│   └── saas_cleaned.csv           # 清洗后（含 searchable_text）
├── notebooks/
│   ├── 01_data_cleaning.ipynb
│   ├── 02_vector_search.ipynb
│   ├── 03_classification.ipynb
│   └── 04_sentiment_or_positioning.ipynb
├── src/
│   └── clean_data.py              # 可复用清洗逻辑
├── gradio_dashboard.py
├── pyproject.toml                 # UV 项目管理
├── uv.lock
├── requirements.txt               # 兼容 pip 导出
└── README.md
```

## 环境（UV）

```bash
# 安装依赖
uv sync

# 数据清洗 → data/saas_cleaned.csv
uv run python -m src.clean_data

# 启动 Dashboard（检索接入前为 UI stub）
uv run python gradio_dashboard.py

# Jupyter
uv run jupyter notebook
```

也可用 `pip install -r requirements.txt`（由 `uv export` 生成）。

## 数据说明

当前源数据约 **5,624** 条、**31** 列工程特征（upvote、launch timing、topic flags、viral 等）。  
源 CSV **没有** 自由文本 `topics` / 官网 URL / PH URL；清洗阶段用 `is_ai_product` 等 flag 合成 `topics` 与 `main_category`。

## 实施优先级

1. **MVP（1–2 天）**: 清洗 → Embedding → Gradio 语义搜索  
2. 接入品类过滤 + 31 列特征排序加权  
3. 分类与定位标签，做成可对外演示的完整产品
