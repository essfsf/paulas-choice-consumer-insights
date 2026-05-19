from pathlib import Path
import re

import jieba
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "paulas_choice_interview_demo.html"


BRAND_COLORS = {
    "green": "#2F6F4E",
    "mint": "#77BFA3",
    "coral": "#E86F5C",
    "amber": "#F2B84B",
    "blue": "#3C78D8",
    "ink": "#263238",
    "muted": "#6B7280",
    "panel": "#F7F9F6",
}

SENTIMENT_LABELS = {
    "positive": "正面",
    "neutral": "中性",
    "negative": "负面",
}

SENTIMENT_COLORS = {
    "positive": BRAND_COLORS["green"],
    "neutral": "#A6ADB4",
    "negative": BRAND_COLORS["coral"],
}

TERM_GROUPS = {
    "水杨酸/BHA": ["水杨酸", "bha", "BHA", "刷酸", "去角质", "黑头", "闭口"],
    "A醇/抗老": ["a醇", "A醇", "视黄醇", "retinol", "抗老", "抗初老"],
    "烟酰胺/提亮": ["烟酰胺", "美白", "提亮", "淡斑", "肤色"],
    "防晒": ["防晒", "spf", "SPF"],
    "痘痘修护": ["痘", "祛痘", "痘印", "粉刺", "闭口"],
    "敏感耐受": ["敏感", "刺痛", "泛红", "过敏", "耐受", "刺激"],
    "肤感质地": ["油腻", "搓泥", "黏", "清爽", "吸收", "肤感", "厚重"],
    "毛孔控油": ["毛孔", "控油", "出油", "油皮", "黑头"],
}

STOPWORDS = {
    "一个", "这个", "真的", "可以", "感觉", "就是", "没有", "不是", "还有", "已经",
    "比较", "一下", "时候", "还是", "什么", "怎么", "因为", "所以", "如果", "但是",
    "姐妹", "哈哈", "哈哈哈", "小红书", "话题", "分享", "宝拉", "珍选", "Paula", "Choice",
}


def parse_count(value):
    if pd.isna(value):
        return 0.0
    text = str(value).strip()
    if text.endswith("万"):
        try:
            return float(text[:-1]) * 10000
        except ValueError:
            return 0.0
    try:
        return float(text)
    except ValueError:
        return 0.0


def load_data():
    contents = pd.read_csv(ROOT / "csv" / "search_contents_2026-04-08.csv")
    comments = pd.read_csv(ROOT / "csv" / "search_comments_2026-04-08.csv")
    absa = pd.read_csv(ROOT / "absa_results.csv")

    contents["liked_count_num"] = contents["liked_count"].apply(parse_count)
    for col in ["collected_count", "comment_count", "share_count"]:
        contents[col] = pd.to_numeric(contents[col], errors="coerce").fillna(0)
    contents["publish_time"] = pd.to_datetime(contents["time"], unit="ms", errors="coerce")
    comments["create_dt"] = pd.to_datetime(comments["create_time"], unit="ms", errors="coerce")
    comments["content"] = comments["content"].fillna("").astype(str)
    absa["comment"] = absa["comment"].fillna("").astype(str)
    absa["sentiment_cn"] = absa["sentiment"].map(SENTIMENT_LABELS)
    return contents, comments, absa


def sentiment_summary(absa):
    summary = (
        absa.groupby(["aspect", "sentiment"])
        .size()
        .reset_index(name="count")
    )
    totals = summary.groupby("aspect")["count"].transform("sum")
    summary["rate"] = summary["count"] / totals
    summary["sentiment_cn"] = summary["sentiment"].map(SENTIMENT_LABELS)
    return summary


def make_kpis(contents, comments, absa, summary):
    negative = summary[summary["sentiment"] == "negative"].sort_values("rate", ascending=False)
    top_pain = negative.iloc[0]
    confident = (absa["confidence"] >= 0.6).mean()
    avg_engagement = (
        contents["liked_count_num"]
        + contents["collected_count"]
        + contents["comment_count"]
        + contents["share_count"]
    ).mean()
    return [
        ("公开笔记", f"{len(contents):,}", "覆盖小红书搜索结果"),
        ("消费者评论", f"{len(comments):,}", "用于文本清洗与分析"),
        ("ABSA 推理", f"{len(absa):,}", "评论 × 成分/功效/肤感"),
        ("首要痛点", f"{top_pain['aspect']} {top_pain['rate']:.1%}", "负面占比最高"),
        ("高置信保留", f"{confident:.1%}", "confidence ≥ 0.60"),
        ("平均互动", f"{avg_engagement:,.0f}", "点赞+收藏+评论+分享"),
    ]


def fig_sentiment_bar(summary):
    fig = px.bar(
        summary,
        x="aspect",
        y="rate",
        color="sentiment",
        text=summary["rate"].map(lambda x: f"{x:.0%}"),
        color_discrete_map=SENTIMENT_COLORS,
        category_orders={"sentiment": ["positive", "neutral", "negative"]},
        labels={"aspect": "", "rate": "占比", "sentiment": "情感"},
        title="成分-功效-肤感：细粒度情感结构",
    )
    fig.update_traces(textposition="inside", hovertemplate="%{x}<br>%{customdata}: %{y:.1%}<extra></extra>")
    for trace in fig.data:
        trace.customdata = [SENTIMENT_LABELS.get(trace.name, trace.name)] * len(trace.x)
        trace.name = SENTIMENT_LABELS.get(trace.name, trace.name)
    fig.update_layout(barmode="stack", yaxis_tickformat=".0%", legend_title_text="")
    return style_fig(fig)


def fig_radar(summary):
    pivot = summary.pivot(index="aspect", columns="sentiment", values="rate").fillna(0)
    categories = ["成分", "功效", "肤感"]
    fig = go.Figure()
    for sentiment in ["positive", "negative", "neutral"]:
        values = [pivot.loc[c, sentiment] if c in pivot.index else 0 for c in categories]
        fig.add_trace(
            go.Scatterpolar(
                r=values + values[:1],
                theta=categories + categories[:1],
                fill="toself",
                name=SENTIMENT_LABELS[sentiment],
                line_color=SENTIMENT_COLORS[sentiment],
                hovertemplate="%{theta}: %{r:.1%}<extra></extra>",
            )
        )
    fig.update_layout(
        title="新品趋势雷达：三维口碑形态",
        polar=dict(radialaxis=dict(tickformat=".0%", range=[0, 0.75])),
        legend_title_text="",
    )
    return style_fig(fig)


def fig_engagement(contents):
    top = contents.copy()
    top["engagement"] = (
        top["liked_count_num"] + top["collected_count"] + top["comment_count"] + top["share_count"]
    )
    top = top.sort_values("engagement", ascending=False).head(12)
    top["short_title"] = top["title"].fillna("无标题").str.slice(0, 22)
    fig = px.bar(
        top.sort_values("engagement"),
        x="engagement",
        y="short_title",
        orientation="h",
        color="type",
        color_discrete_map={"normal": BRAND_COLORS["mint"], "video": BRAND_COLORS["amber"]},
        labels={"engagement": "总互动", "short_title": "", "type": "内容类型"},
        title="高互动内容：消费者最愿意讨论什么",
        hover_data={"title": True, "liked_count_num": ":,.0f", "collected_count": ":,.0f", "comment_count": ":,.0f"},
    )
    fig.update_layout(legend_title_text="")
    return style_fig(fig)


def fig_geo(comments):
    geo = comments["ip_location"].dropna().value_counts().head(15).reset_index()
    geo.columns = ["地区", "评论数"]
    fig = px.bar(
        geo,
        x="地区",
        y="评论数",
        color="评论数",
        color_continuous_scale=["#DCEFE6", BRAND_COLORS["green"]],
        title="评论地域分布：潜在种草和转化市场",
    )
    fig.update_layout(coloraxis_showscale=False)
    return style_fig(fig)


def term_metrics(comments, absa):
    comment_sent = (
        absa.groupby("comment")
        .agg(
            negative_rate=("sentiment", lambda s: (s == "negative").mean()),
            positive_rate=("sentiment", lambda s: (s == "positive").mean()),
            confidence=("confidence", "mean"),
        )
        .reset_index()
    )
    rows = []
    all_comments = comments[["content", "like_count"]].copy()
    all_comments["like_count"] = pd.to_numeric(all_comments["like_count"], errors="coerce").fillna(0)

    for group, terms in TERM_GROUPS.items():
        pattern = "|".join(map(re.escape, terms))
        matched = all_comments[all_comments["content"].str.contains(pattern, case=False, regex=True, na=False)]
        sent = comment_sent[comment_sent["comment"].str.contains(pattern, case=False, regex=True, na=False)]
        if len(matched) == 0:
            continue
        rows.append(
            {
                "机会主题": group,
                "声量": len(matched),
                "互动权重": matched["like_count"].sum(),
                "负面率": sent["negative_rate"].mean() if len(sent) else 0,
                "正面率": sent["positive_rate"].mean() if len(sent) else 0,
                "平均置信": sent["confidence"].mean() if len(sent) else 0,
            }
        )
    return pd.DataFrame(rows)


def fig_opportunity_matrix(metrics):
    fig = px.scatter(
        metrics,
        x="声量",
        y="负面率",
        size="互动权重",
        color="正面率",
        text="机会主题",
        color_continuous_scale=["#E86F5C", "#F2B84B", "#2F6F4E"],
        size_max=56,
        labels={"负面率": "痛点强度", "声量": "消费者声量", "正面率": "正面率"},
        title="新品机会矩阵：高声量 × 高痛点优先验证",
        hover_data={"互动权重": ":,.0f", "平均置信": ":.2f"},
    )
    fig.update_traces(textposition="top center")
    fig.update_yaxes(tickformat=".0%")
    return style_fig(fig)


def fig_keyword_rank(comments):
    text = " ".join(comments["content"].dropna().astype(str).tolist())
    words = []
    for token in jieba.cut(text):
        token = token.strip()
        if len(token) < 2:
            continue
        if token in STOPWORDS:
            continue
        if re.search(r"^[a-zA-Z0-9_]+$", token) and len(token) < 3:
            continue
        words.append(token)
    freq = pd.Series(words).value_counts().head(22).reset_index()
    freq.columns = ["关键词", "出现次数"]
    fig = px.bar(
        freq.sort_values("出现次数"),
        x="出现次数",
        y="关键词",
        orientation="h",
        color="出现次数",
        color_continuous_scale=["#DCEFE6", BRAND_COLORS["green"]],
        title="消费者原话高频词：从评论里提炼洞察语言",
    )
    fig.update_layout(coloraxis_showscale=False)
    return style_fig(fig)


def fig_timeline(contents, comments):
    posts = contents.dropna(subset=["publish_time"]).copy()
    comm = comments.dropna(subset=["create_dt"]).copy()
    post_month = posts.groupby(posts["publish_time"].dt.to_period("M")).size().reset_index(name="笔记数")
    comment_month = comm.groupby(comm["create_dt"].dt.to_period("M")).size().reset_index(name="评论数")
    post_month["month"] = post_month["publish_time"].astype(str)
    comment_month["month"] = comment_month["create_dt"].astype(str)
    merged = pd.merge(post_month[["month", "笔记数"]], comment_month[["month", "评论数"]], on="month", how="outer").fillna(0)
    merged = merged.sort_values("month").tail(18)

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_bar(x=merged["month"], y=merged["笔记数"], name="笔记数", marker_color=BRAND_COLORS["mint"])
    fig.add_trace(
        go.Scatter(x=merged["month"], y=merged["评论数"], mode="lines+markers", name="评论数", line_color=BRAND_COLORS["coral"]),
        secondary_y=True,
    )
    fig.update_layout(title="趋势监测：内容供给与消费者讨论节奏", legend_title_text="")
    fig.update_yaxes(title_text="笔记数", secondary_y=False)
    fig.update_yaxes(title_text="评论数", secondary_y=True)
    return style_fig(fig)


def fig_content_mix(contents):
    source = contents["source_keyword"].value_counts().reset_index()
    source.columns = ["搜索词", "笔记数"]
    ctype = contents["type"].value_counts().reset_index()
    ctype.columns = ["类型", "笔记数"]

    fig = make_subplots(
        rows=1,
        cols=2,
        specs=[[{"type": "domain"}, {"type": "domain"}]],
        subplot_titles=("搜索入口", "内容形态"),
    )
    fig.add_trace(go.Pie(labels=source["搜索词"], values=source["笔记数"], hole=0.55, marker_colors=px.colors.qualitative.Set2), 1, 1)
    fig.add_trace(go.Pie(labels=ctype["类型"], values=ctype["笔记数"], hole=0.55, marker_colors=[BRAND_COLORS["mint"], BRAND_COLORS["amber"]]), 1, 2)
    fig.update_layout(title="数据来源结构：可解释的采集口径", showlegend=True)
    return style_fig(fig)


def style_fig(fig):
    fig.update_layout(
        template="plotly_white",
        font=dict(family="Arial, PingFang SC, Heiti SC, Microsoft YaHei, sans-serif", color=BRAND_COLORS["ink"]),
        title=dict(font=dict(size=19), x=0.02),
        paper_bgcolor="white",
        plot_bgcolor="white",
        margin=dict(l=42, r=28, t=70, b=42),
        hoverlabel=dict(bgcolor="white", font_size=13),
    )
    return fig


def fig_to_div(fig, include_plotlyjs=False):
    return pio.to_html(
        fig,
        include_plotlyjs=include_plotlyjs,
        full_html=False,
        config={"displaylogo": False, "responsive": True},
    )


def build_html():
    contents, comments, absa = load_data()
    summary = sentiment_summary(absa)
    metrics = term_metrics(comments, absa)
    kpis = make_kpis(contents, comments, absa, summary)

    figures = [
        fig_sentiment_bar(summary),
        fig_radar(summary),
        fig_opportunity_matrix(metrics),
        fig_keyword_rank(comments),
        fig_engagement(contents),
        fig_geo(comments),
        fig_timeline(contents, comments),
        fig_content_mix(contents),
    ]

    kpi_cards = "\n".join(
        f"""
        <div class="kpi-card">
          <div class="kpi-label">{label}</div>
          <div class="kpi-value">{value}</div>
          <div class="kpi-note">{note}</div>
        </div>
        """
        for label, value, note in kpis
    )
    fig_divs = "\n".join(
        f'<section class="chart-card">{fig_to_div(fig, include_plotlyjs=(idx == 0))}</section>'
        for idx, fig in enumerate(figures)
    )

    html = f"""
<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Paula's Choice 新品趋势雷达 Demo</title>
  <style>
    :root {{
      --green: {BRAND_COLORS["green"]};
      --mint: {BRAND_COLORS["mint"]};
      --coral: {BRAND_COLORS["coral"]};
      --amber: {BRAND_COLORS["amber"]};
      --ink: {BRAND_COLORS["ink"]};
      --muted: {BRAND_COLORS["muted"]};
      --panel: {BRAND_COLORS["panel"]};
      --line: #DFE7DE;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: #FFFFFF;
      color: var(--ink);
      font-family: Arial, "PingFang SC", "Heiti SC", "Microsoft YaHei", sans-serif;
    }}
    header {{
      padding: 34px 42px 24px;
      background: linear-gradient(180deg, #F1F7F1 0%, #FFFFFF 100%);
      border-bottom: 1px solid var(--line);
    }}
    .eyebrow {{
      margin: 0 0 8px;
      color: var(--green);
      font-weight: 700;
      letter-spacing: 0;
      font-size: 14px;
    }}
    h1 {{
      margin: 0;
      font-size: 34px;
      line-height: 1.16;
      letter-spacing: 0;
    }}
    .subtitle {{
      margin: 12px 0 0;
      max-width: 980px;
      color: var(--muted);
      font-size: 16px;
      line-height: 1.65;
    }}
    .main {{
      padding: 26px 42px 42px;
      max-width: 1480px;
      margin: 0 auto;
    }}
    .kpi-grid {{
      display: grid;
      grid-template-columns: repeat(6, minmax(0, 1fr));
      gap: 14px;
      margin-bottom: 22px;
    }}
    .kpi-card {{
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 16px 16px 14px;
      background: var(--panel);
      min-height: 116px;
    }}
    .kpi-label {{
      color: var(--muted);
      font-size: 13px;
      margin-bottom: 8px;
    }}
    .kpi-value {{
      color: var(--green);
      font-size: 27px;
      font-weight: 800;
      line-height: 1.1;
      margin-bottom: 8px;
    }}
    .kpi-note {{
      color: var(--muted);
      font-size: 12px;
      line-height: 1.45;
    }}
    .story {{
      display: grid;
      grid-template-columns: 1.15fr 0.85fr;
      gap: 18px;
      margin-bottom: 22px;
    }}
    .insight-card {{
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 20px;
      background: #FFFFFF;
    }}
    .insight-card h2 {{
      margin: 0 0 12px;
      font-size: 20px;
      letter-spacing: 0;
    }}
    .insight-card p, .insight-card li {{
      color: var(--muted);
      font-size: 14px;
      line-height: 1.7;
    }}
    .insight-card ul {{
      margin: 0;
      padding-left: 20px;
    }}
    .chart-grid {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 18px;
    }}
    .chart-card {{
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #FFFFFF;
      padding: 8px;
      min-height: 430px;
    }}
    .chart-card .plotly-graph-div {{
      width: 100% !important;
    }}
    footer {{
      color: var(--muted);
      font-size: 12px;
      padding: 8px 0 0;
    }}
    @media (max-width: 1100px) {{
      header {{ padding: 28px 20px 20px; }}
      .main {{ padding: 20px; }}
      .kpi-grid {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
      .story, .chart-grid {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <header>
    <p class="eyebrow">Interview Demo · Digital Consumer Insights</p>
    <h1>Paula's Choice 新品趋势雷达</h1>
    <p class="subtitle">
      基于小红书公开笔记与评论数据，串联数据采集、清洗、BERT 情感识别、ABSA 维度拆解和商业机会识别，
      对齐 JD 中“实时分析消费者评价与趋势、输出新品开发和营销建议”的项目目标。
    </p>
  </header>
  <main class="main">
    <section class="kpi-grid">
      {kpi_cards}
    </section>

    <section class="story">
      <div class="insight-card">
        <h2>面试讲述主线</h2>
        <ul>
          <li>先证明数据体系：搜索词、内容类型、地域分布和互动指标都可追踪。</li>
          <li>再证明模型能力：把每条评论拆成“成分、功效、肤感”三个 aspect 做细粒度情感。</li>
          <li>最后落到业务动作：用“声量 × 负面率 × 互动权重”识别新品机会和营销沟通重点。</li>
        </ul>
      </div>
      <div class="insight-card">
        <h2>核心洞察</h2>
        <p>
          成分与功效整体口碑较正向，但肤感负面率明显更高。面试时可以把它解释为：
          消费者认可功效逻辑，但产品体验仍影响复购和种草效率，因此新品策略应同时验证
          “有效成分 + 低刺激 + 清爽肤感”的组合。
        </p>
      </div>
    </section>

    <section class="chart-grid">
      {fig_divs}
    </section>
    <footer>
      Data: Xiaohongshu public search/comment exports · Model output: ABSA sentiment predictions · Built with Plotly
    </footer>
  </main>
</body>
</html>
"""
    OUTPUT.write_text(html, encoding="utf-8")
    return OUTPUT


if __name__ == "__main__":
    output = build_html()
    print(f"Demo generated: {output}")
