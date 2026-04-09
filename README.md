# Paula's Choice Consumer Insights — Xiaohongshu NLP Analysis

Personal project aligned with Unilever U2 Digital Consumer Insights Track.

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

## Pipeline
