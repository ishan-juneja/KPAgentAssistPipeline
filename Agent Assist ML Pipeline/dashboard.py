# ---------------------------------------------------------------------------------
# HR DASHBOARD v2  ·  All–in–one Streamlit script (dashboard_2/app.py)
# ---------------------------------------------------------------------------------
# 1.  To-Do/Build Checklist (auto-generated)
# • Load HR conversation CSV(s) – default to sample_data.csv if none supplied.
# • Provide global sidebar filters (Parent Category, Agent, Date range).
# • Use Instrument Sans font + navy/yellow brand colours.
# • Render 11 high-value charts inside collapsible expanders:
#     1  Category Volume vs Error Rate              (bar + line)
#     2  Sub-Topic Confusion Treemap                (treemap)
#     3  Agent Workload vs Error                    (scatter)
#     4  Temporal Error Heat-map                    (heatmap)
#     5  Conversation Length vs Error               (scatter + trend)
#     6  Positive-Feedback Rate by Category         (bar)
#     7  Duplicate Query Frequency                  (bar)
#     8  Low-Confidence Outlier Tracker             (table)
#     9  Parent ↔ Sub-Topic Mismatch Sunburst       (sunburst)
#    10  Knowledge Article Impact                   (violin/box)
#    11  Error Similarity Trend Over Time           (line)
# • Each chart ends with a placeholder insight line.
# • Uses Plotly template = "simple_white", colour palette navy (#001f3f) & yellow (#FFDC00).
# • Entire app lives in this single file; sample dataset stored at ../dashboard_2/sample_data.csv.
# ---------------------------------------------------------------------------------

import os
import glob
import textwrap
from datetime import datetime
from pathlib import Path

# Base libs
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# Optional dependency check for statsmodels (required for Plotly trendlines)
try:
    import statsmodels.api as _sm  # noqa: F401
    _HAS_STATSMODELS = True
except ModuleNotFoundError:
    _HAS_STATSMODELS = False

# -----------------------------------------------------------------------------
# Constants & Theme
# -----------------------------------------------------------------------------
# Rework colour scheme: primary bright yellow, secondary black
PRIMARY = "#FFDC00"  # bright yellow for bars/points
SECONDARY = "#000000"  # black for contrast lines/alt series
FONT_FAMILY = "Instrument Sans, sans-serif"

st.set_page_config(page_title="HR Query Analytics", layout="wide")

# Inject basic CSS to register custom font if available --------------------------------
# Looks for InstrumentSans.ttf in project root.
font_path = Path(__file__).resolve().parent.parent / "InstrumentSans.ttf"
if font_path.exists():
    st.markdown(
        f"""
        <style>
        @font-face {{
            font-family: 'Instrument Sans';
            src: url('file://{font_path}') format('truetype');
        }}
        html, body, [class*="css"]  {{
            font-family: 'Instrument Sans', sans-serif;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )

# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------

def primary_bar(x, y, name=""):
    return go.Bar(x=x, y=y, name=name or "", marker_color=PRIMARY)


def secondary_line(x, y, name=""):
    return go.Scatter(x=x, y=y, name=name or "", mode="lines+markers", line=dict(color=SECONDARY, width=3))


# -----------------------------------------------------------------------------
# Data Loading
# -----------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def load_data(uploaded_files):
    """Read uploaded CSV(s) or fall back to bundled sample."""
    if uploaded_files:
        dfs = [pd.read_csv(f) for f in uploaded_files]
    else:
        sample_path = Path(__file__).parent / "Agent_Assist_Final_Labeled_Data_3.csv"
        dfs = [pd.read_csv(sample_path)]
    df = pd.concat(dfs, ignore_index=True)

    # Type casts
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
    int_cols = ["hour_of_day", "conversation_length"]
    for col in int_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
    float_cols = [
        "Parent Error Similarity Score",
        "Parent Category Similarity_Score",
        "SubCategory Similarity_Score",
    ]
    for col in float_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Ensure helper cols exist
    if "day_of_week" not in df.columns:
        df["day_of_week"] = df["Timestamp"].dt.day_name()
    if "hour_of_day" not in df.columns:
        df["hour_of_day"] = df["Timestamp"].dt.hour

    return df


# -----------------------------------------------------------------------------
# Sidebar – file upload & global filters (DATA SOURCE SECTION)
# -----------------------------------------------------------------------------
with st.sidebar:
    st.title("Controls")
    uploaded_files = st.file_uploader(
        "Upload one or more weekly CSV exports",
        type="csv",
        accept_multiple_files=True,
    )

    df_full = load_data(uploaded_files)

    # Filters -----------------------------------------------------
    parent_options = sorted(df_full["Parent Category Topic"].dropna().unique())
    agent_options = sorted(df_full["Agent_ID"].dropna().unique())

    parent_filter = st.multiselect("Parent Category", parent_options)
    agent_filter = st.multiselect("Agent", agent_options)

    # Date range filter
    # Guard against empty or NaT
    min_ts, max_ts = df_full["Timestamp"].min(), df_full["Timestamp"].max()
    if pd.isna(min_ts) or pd.isna(max_ts):
        min_ts = max_ts = pd.Timestamp.today()

    date_range = st.date_input(
        "Date range",
        value=(min_ts.date(), max_ts.date()),
        min_value=min_ts.date(),
        max_value=max_ts.date(),
    )

    threshold = st.slider(
        "High-confusion threshold (Parent Error Similarity)",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.05,
    )

    # Feedback sentiment filter
    feedback_options = sorted(df_full["Feedback"].dropna().unique())
    feedback_filter = st.multiselect("Feedback Type", feedback_options)


# Apply filter mask -----------------------------------------------------------
mask = pd.Series(True, index=df_full.index)
if parent_filter:
    mask &= df_full["Parent Category Topic"].isin(parent_filter)
if agent_filter:
    mask &= df_full["Agent_ID"].isin(agent_filter)
if feedback_filter:
    mask &= df_full["Feedback"].isin(feedback_filter)
start_date, end_date = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
mask &= df_full["Timestamp"].between(start_date, end_date + pd.Timedelta(days=1))

data = df_full[mask].copy()

# -----------------------------------------------------------------------------
# Page Title
# -----------------------------------------------------------------------------
st.title("HR Conversation Analytics Dashboard (v2)")
st.caption(
    "Interactive insights for HR Agent-Assist performance.  "
    "Colours: Navy = volume/context, Yellow = confusion/alert.  "
)

# -----------------------------------------------------------------------------
# Utility: add placeholder insight line
# -----------------------------------------------------------------------------

def add_insight(text: str):
    st.markdown(f"*Insight ⇢ {text}*")

cat_agg = (
    data["Parent Category Topic"]
    .value_counts()
    .reset_index()
)
cat_agg.columns = ["Parent Category Topic", "Volume"]   # <= explicit names

sub_agg = (
    data["SubCategory Topic"]
    .value_counts()
    # .head(st.slider("Top N sub-topics", 5, 50, 20))
    .reset_index()
)
sub_agg.columns = ["SubCategory Topic", "Volume"]       # <= explicit names

# ──────────────────────────────────────────────────────────────────────────────
#  CHARTS  —  similarity-free, version-safe
# ──────────────────────────────────────────────────────────────────────────────

# 1 · Query Volume by Parent Category
with st.expander("1 · Query Volume by Parent Category"):
    cat_agg = (
        data["Parent Category Topic"]
        .value_counts()
        .reset_index()
    )
    cat_agg.columns = ["Parent Category Topic", "Volume"]  # explicit names

    fig = px.bar(
        cat_agg,
        x="Parent Category Topic",
        y="Volume",
        text_auto=True,
        color_discrete_sequence=[PRIMARY],
        title="Total Queries by Parent Category",
    )
    fig.update_layout(
        xaxis_tickangle=-45,
        template="simple_white",
        font=dict(family=FONT_FAMILY),
    )
    st.plotly_chart(fig, use_container_width=True)
    add_insight("Shows which HR areas drive the most support demand.")

# 10 · Positive-Feedback Rate by Sub-Category
with st.expander("1B · Positive-Feedback Rate by Sub-Category"):
    top_n_sub_fb = st.slider("Top N sub-topics", 5, 50, 20, key="sub_fb_slider")

    sub_fb = (
        data.groupby(["SubCategory Topic", "Feedback"])
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )
    if "positive" not in sub_fb.columns:
        sub_fb["positive"] = 0
    sub_fb["Positive Rate"] = sub_fb["positive"] / sub_fb.drop(columns=["SubCategory Topic"]).sum(axis=1)

    # Show the most-answered sub-topics to keep axis legible
    sub_fb = (
        sub_fb
        .sort_values("positive", ascending=False)
        .head(top_n_sub_fb)
    )

    fig = px.bar(
        sub_fb,
        y="SubCategory Topic",
        x="Positive Rate",
        orientation="h",
        color_discrete_sequence=[PRIMARY],
        text_auto=".0%",
        title=f"Positive Feedback % for Top {top_n_sub_fb} Sub-Topics",
    )
    fig.update_layout(
        xaxis_tickformat=".0%",
        yaxis=dict(autorange="reversed"),
        template="simple_white",
        font=dict(family=FONT_FAMILY),
        height=600
    )
    st.plotly_chart(fig, use_container_width=True)
    add_insight("Pinpoints sub-topics where users remain unhappy despite resolutions.")


# 2 · Top Sub-Topics by Volume
with st.expander("2 · Top Sub-Topics by Volume"):
    top_n = st.slider("Top N sub-topics", 5, 50, 20, key="subslider_expander")
    sub_agg = (
        data["SubCategory Topic"]
        .value_counts()
        .head(top_n)
        .reset_index()
    )
    sub_agg.columns = ["SubCategory Topic", "Volume"]

    fig = px.bar(
        sub_agg,
        y="SubCategory Topic",
        x="Volume",
        orientation="h",
        color_discrete_sequence=[PRIMARY],
        title=f"Most Frequent Sub-Topics (Top {top_n})",
    )
    fig.update_layout(
        yaxis=dict(autorange="reversed"),
        template="simple_white",
        font=dict(family=FONT_FAMILY),
    )
    st.plotly_chart(fig, use_container_width=True)
    add_insight("Identifies granular themes driving support demand.")


# 3 · Positive-Feedback Rate by Category
with st.expander("3 · Positive-Feedback Rate by Category"):
    fb = (
        data.groupby(["Parent Category Topic", "Feedback"])
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )
    if "positive" not in fb.columns:
        fb["positive"] = 0
    fb["Positive Rate"] = fb["positive"] / fb.drop(columns=["Parent Category Topic"]).sum(axis=1)

    fig = px.bar(
        fb,
        x="Parent Category Topic",
        y="Positive Rate",
        color_discrete_sequence=[PRIMARY],
        text_auto=".0%",
        title="Positive Feedback % per Category",
    )
    fig.update_layout(
        yaxis_tickformat=".0%",
        xaxis_tickangle=-45,
        template="simple_white",
        font=dict(family=FONT_FAMILY),
    )
    st.plotly_chart(fig, use_container_width=True)
    add_insight("Low rates highlight categories where users remain dissatisfied despite resolution.")


# 4 · Daily Query Volume Trend
with st.expander("4 · Daily Query Volume Trend"):
    daily = (
        data.set_index("Timestamp")
        .resample("D")
        .size()
        .reset_index(name="Volume")
    )
    fig = px.line(
        daily,
        x="Timestamp",
        y="Volume",
        markers=True,
        color_discrete_sequence=[PRIMARY],
        title="Daily Query Volume",
    )
    fig.update_layout(template="simple_white", font=dict(family=FONT_FAMILY))
    st.plotly_chart(fig, use_container_width=True)
    add_insight("Spikes often align with policy releases, system outages, or payday cycles.")


# 5 · Agent Positive-Feedback Ratio
with st.expander("5 · Agent Positive-Feedback Ratio"):
    agent_fb = (
        data.groupby(["Agent_ID", "Feedback"]).size().unstack(fill_value=0).reset_index()
    )
    if "positive" not in agent_fb.columns:
        agent_fb["positive"] = 0
    agent_fb["Positive Rate"] = agent_fb["positive"] / agent_fb.drop(columns=["Agent_ID"]).sum(axis=1)

    total_queries = agent_fb.drop(columns=["Agent_ID", "Positive Rate"]).sum(axis=1)

    fig = px.scatter(
        agent_fb,
        x="positive",
        y="Positive Rate",
        size=total_queries,
        text="Agent_ID",
        color_discrete_sequence=[PRIMARY],
        labels={"positive": "# Positive", "Positive Rate": "Positive Rate"},
        title="Agent Performance (bubble ≈ total queries)",
    )
    fig.update_traces(textfont=dict(color="black", family=FONT_FAMILY))
    fig.update_layout(template="simple_white", font=dict(family=FONT_FAMILY))
    st.plotly_chart(fig, use_container_width=True)
    add_insight("Bottom-left bubbles = high volume but low satisfaction → coaching needed.")


# 6 · Conversation Length Distribution
with st.expander("6 · Conversation Length Distribution"):
    box_df = data[["Parent Category Topic", "conversation_length"]].dropna()
    fig = px.box(
        box_df,
        x="Parent Category Topic",
        y="conversation_length",
        points="outliers",
        color_discrete_sequence=[PRIMARY],
        title="Conversation Length per Category",
    )
    fig.update_layout(
        yaxis_title="Seconds",
        xaxis_tickangle=-45,
        template="simple_white",
        font=dict(family=FONT_FAMILY),
    )
    st.plotly_chart(fig, use_container_width=True)
    add_insight("Long-tail outliers often mark complex policy questions or broken workflows.")


# 7 · Top Duplicate Queries
with st.expander("7 · Top Duplicate Queries"):
    dup_counts = (
        data["Knowledge_Answer"]
        .fillna("-")
        .value_counts()
        .reset_index()
    )
    dup_counts.columns = ["Knowledge_Answer", "Count"]

    top_n_dup = st.slider("Show top-N duplicates", 5, 50, 15, key="dup_slider")
    dup_top = dup_counts.head(top_n_dup)

    fig = px.bar(
        dup_top,
        y="Knowledge_Answer",
        x="Count",
        orientation="h",
        color_discrete_sequence=[PRIMARY],
        title=f"Top {top_n_dup} Repeated Knowledge Answers",
    )
    fig.update_layout(
        yaxis=dict(autorange="reversed"),
        template="simple_white",
        font=dict(family=FONT_FAMILY),
        height=600,
    )
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(dup_top, use_container_width=True, hide_index=True)
    add_insight("High-volume duplicates are candidates for self-serve FAQs or macro responses.")


# 8 · Topic Seasonality Heat-map
with st.expander("8 · Topic Seasonality Heat-map"):
    month_topic = (
        data.groupby([data["Timestamp"].dt.to_period("M").astype(str), "Parent Category Topic"])
        .size()
        .reset_index(name="Volume")
    )
    fig = px.density_heatmap(
        month_topic,
        x="Timestamp",
        y="Parent Category Topic",
        z="Volume",
        color_continuous_scale=[[0, "#E8E8E8"], [1, PRIMARY]],
        title="Monthly Volume Heat-map by Category",
    )
    fig.update_layout(template="simple_white", font=dict(family=FONT_FAMILY))
    st.plotly_chart(fig, use_container_width=True)
    add_insight("Seasonal surges (e.g., open enrollment) pop visually for proactive planning.")


# 9 · Error-Category Crosstab  (if labels exist)
with st.expander("9 · Error-Category Crosstab"):
    if "Parent Error Topic" not in data.columns or data["Parent Error Topic"].isna().all():
        st.info("No 'Parent Error Topic' labels in the current dataset.")
    else:
        pivot = (
            data.pivot_table(
                index="Parent Category Topic",
                columns="Parent Error Topic",
                values="Agent_ID",
                aggfunc="count",
                fill_value=0,
            )
            .astype(int)
        )
        st.dataframe(pivot, use_container_width=True)
        heat_df = pivot.reset_index().melt(id_vars="Parent Category Topic", var_name="Error", value_name="Count")
        fig = px.density_heatmap(
            heat_df,
            x="Error",
            y="Parent Category Topic",
            z="Count",
            color_continuous_scale=[[0, "#E8E8E8"], [1, PRIMARY]],
            title="Error Categories by Parent Topic",
        )
        fig.update_layout(
            xaxis_tickangle=-45,
            template="simple_white",
            font=dict(family=FONT_FAMILY),
            height=500,
        )
        st.plotly_chart(fig, use_container_width=True)
        add_insight("Dark cells reveal error themes concentrated in specific HR areas.")

# 10 · Error-Topic vs Sub-Topic Heat-map
with st.expander("10 · Error Topics within Top Sub-Topics"):
    if "Parent Error Topic" not in data.columns or data["Parent Error Topic"].isna().all():
        st.info("No 'Parent Error Topic' labels in the current dataset.")
    else:
        # pick same N as chart 10 for consistency
        top_sub_err = (
            data["SubCategory Topic"]
            .value_counts()
            .head(top_n_sub_fb)      # reuse slider value
            .index
        )

        err_sub = (
            data[data["SubCategory Topic"].isin(top_sub_err)]
            .pivot_table(
                index="SubCategory Topic",
                columns="Parent Error Topic",
                values="Agent_ID",
                aggfunc="count",
                fill_value=0,
            )
        )
        hm_df = (
            err_sub
            .reset_index()
            .melt(id_vars="SubCategory Topic", var_name="Error Topic", value_name="Count")
        )

        fig = px.density_heatmap(
            hm_df,
            x="Error Topic",
            y="SubCategory Topic",
            z="Count",
            color_continuous_scale=[[0, "#E8E8E8"], [1, PRIMARY]],
            title="Error Topic Frequency in Top Sub-Topics",
        )
        fig.update_layout(
            xaxis_tickangle=-45,
            template="simple_white",
            font=dict(family=FONT_FAMILY),
            height=600,
        )
        st.plotly_chart(fig, use_container_width=True)
        add_insight("Bright cells reveal which error themes dominate specific sub-topics (e.g., garnishment docs).")


# -----------------------------------------------------------------------------
# Footer / Download filtered data
# -----------------------------------------------------------------------------
st.markdown("---")
st.download_button(
    label="Download filtered data as CSV",
    data=data.to_csv(index=False).encode("utf-8"),
    file_name="filtered_hr_data.csv",
    mime="text/csv",
)


st.caption("© 2025 KP Analytics") 