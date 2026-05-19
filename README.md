# Paula's Choice Consumer Insights

Interactive consumer insights demo for Paula's Choice China market analysis.

## Demo

Open the standalone dashboard:

[paulas_choice_interview_demo.html](./paulas_choice_interview_demo.html)

For GitHub Pages, publish the `/docs` folder. The dashboard entry point is:

[docs/index.html](./docs/index.html)

This demo is designed for HR/interview review. It can be opened directly in a browser and does not require a running Python server.

## Project Overview
End-to-end NLP pipeline analysing Paula's Choice consumer sentiment on Xiaohongshu,
covering data collection → preprocessing → sentiment analysis → aspect-based insights.

## Key Results
| Aspect | Positive Rate | Negative Rate | Sample Size |
|--------|--------------|---------------|-------------|
| 成分 (Ingredients) | 69.4% | 24.8% | 900 |
| 功效 (Efficacy)    | 68.0% | 27.0% | 893 |
| 肤感 (Skin Feel)   | 51.0% | 41.5% | 698 |

**Key Insight:** Skin feel is the primary consumer pain point (negative rate 41.5%),
driven by tolerance (耐受) and texture concerns — actionable signal for product iteration.

## Technical Stack
- **Data Collection:** MediaCrawler (Playwright-based), 118 posts + 1,914 comments
- **Preprocessing:** pandas, jieba (custom skincare lexicon), emoji noise removal
- **Sentiment Analysis:** distilbert-base-multilingual-cased-sentiments-student (HuggingFace)
- **ABSA Method:** Aspect-prompt injection across 3 dimensions (5,742 inference calls)
- **Confidence Filtering:** threshold=0.6, retention rate=43.4%
- **Keyword Analysis:** jieba + Counter, stopword-filtered
- **Visualization:** Plotly standalone HTML, optional Dash app

## Run Locally

Regenerate the standalone Plotly HTML:

```bash
python interview_demo.py
```

If using Anaconda:

```bash
/opt/anaconda3/bin/python interview_demo.py
```

Optional Dash app:

```bash
python -m pip install -r requirements.txt
python dash_app.py
```

Then open:

```text
http://127.0.0.1:8050/
```

## Files

| File | Purpose |
|---|---|
| `paulas_choice_interview_demo.html` | Standalone HR/interview demo |
| `docs/index.html` | GitHub Pages entry point |
| `interview_demo.py` | Generates the standalone Plotly HTML |
| `dash_app.py` | Optional Dash web app |
| `absa_results.csv` | Aspect-based sentiment output |
| `csv/` | Source post and comment data |
| `HR_DEMO_GUIDE.md` | Sharing and presentation guide |
