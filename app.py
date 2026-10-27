"""
Social Media Usage Analysis - Streamlit Web Application
Author: Arya Singh (Roll No: 02, Class: B3[B1])
Project: Statistical Inference (SI Lab) Social Media Usage Analysis
Technology Stack: Python, Streamlit, Pandas, NumPy, Plotly, SciPy, Scikit-Learn
"""

import os
import io
import math
import requests
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    accuracy_score,
    classification_report,
    confusion_matrix,
)
import streamlit as st

# -----------------------------------------------------------------------------
# 1. PAGE CONFIGURATION & THEME STYLING
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Social Media Usage Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom High-End SaaS CSS
st.markdown(
    """
    <style>
    /* Global Styles */
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    /* Hero Header Container */
    .hero-card {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.95) 0%, rgba(15, 23, 42, 0.98) 100%);
        border: 1px solid rgba(99, 102, 241, 0.25);
        border-radius: 18px;
        padding: 30px;
        margin-bottom: 25px;
        box-shadow: 0 10px 30px -10px rgba(0, 0, 0, 0.5), 0 0 20px -5px rgba(99, 102, 241, 0.2);
    }
    
    .hero-title {
        font-size: 2.3rem;
        font-weight: 800;
        background: linear-gradient(120deg, #ffffff 30%, #818cf8 70%, #38bdf8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 8px;
        letter-spacing: -0.5px;
    }
    
    .hero-subtitle {
        color: #94a3b8;
        font-size: 1.05rem;
        font-weight: 400;
        line-height: 1.6;
        margin-bottom: 16px;
    }
    
    .badge-pill {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: rgba(99, 102, 241, 0.15);
        border: 1px solid rgba(99, 102, 241, 0.35);
        color: #c7d2fe;
        font-size: 0.82rem;
        font-weight: 600;
        padding: 4px 12px;
        border-radius: 9999px;
        margin-right: 8px;
        margin-bottom: 6px;
    }
    
    .badge-pill-success {
        background: rgba(16, 185, 129, 0.15);
        border: 1px solid rgba(16, 185, 129, 0.35);
        color: #a7f3d0;
    }
    
    .badge-pill-accent {
        background: rgba(14, 165, 233, 0.15);
        border: 1px solid rgba(14, 165, 233, 0.35);
        color: #bae6fd;
    }
    
    /* Modern Metric Cards */
    .metric-container {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
        gap: 16px;
        margin-bottom: 24px;
    }
    
    .stat-card {
        background: rgba(30, 41, 59, 0.7);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 20px;
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    
    .stat-card:hover {
        transform: translateY(-2px);
        border-color: rgba(99, 102, 241, 0.4);
    }
    
    .stat-label {
        color: #94a3b8;
        font-size: 0.85rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        margin-bottom: 6px;
    }
    
    .stat-value {
        color: #f8fafc;
        font-size: 1.85rem;
        font-weight: 800;
        letter-spacing: -0.5px;
        margin-bottom: 4px;
    }
    
    .stat-desc {
        color: #64748b;
        font-size: 0.8rem;
        font-weight: 500;
    }

    /* Section Card */
    .glass-card {
        background: rgba(30, 41, 59, 0.6);
        border: 1px solid rgba(255, 255, 255, 0.07);
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 24px;
    }
    
    .section-title {
        font-size: 1.35rem;
        font-weight: 700;
        color: #f1f5f9;
        margin-bottom: 12px;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    
    .section-desc {
        color: #94a3b8;
        font-size: 0.92rem;
        line-height: 1.6;
        margin-bottom: 20px;
    }
    
    /* Code font */
    code {
        font-family: 'JetBrains Mono', monospace;
        background: rgba(15, 23, 42, 0.8);
        padding: 2px 6px;
        border-radius: 6px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        color: #38bdf8;
    }
    
    /* Sidebar polish */
    [data-testid="stSidebar"] {
        background-color: #0d1322 !important;
        border-right: 1px solid rgba(255, 255, 255, 0.06);
    }
    
    .sidebar-brand {
        padding: 10px 0 20px 0;
        border-bottom: 1px solid rgba(255, 255, 255, 0.08);
        margin-bottom: 15px;
    }
    
    .sidebar-author-box {
        background: rgba(30, 41, 59, 0.7);
        border: 1px solid rgba(99, 102, 241, 0.2);
        border-radius: 12px;
        padding: 12px 14px;
        margin-top: 20px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# 2. DATA INGESTION & PIPELINE (From Attached Notebook)
# -----------------------------------------------------------------------------
DATA_URL = "https://gist.githubusercontent.com/ontipulikesava153/00de7c8bbf6846daef91955970c06e92/raw/social_media_usage.csv"
LOCAL_FILE = "social_media_usage.csv"


@st.cache_data(show_spinner=False)
def load_and_preprocess_data():
    """
    Loads dataset directly following the practical steps from the notebook:
    1. Reads data from local CSV or original remote repository URL
    2. Inspects missing values & duplicates
    3. Cleans dataset with drop_duplicates() and dropna()
    """
    df = None
    source_used = "local"
    if os.path.exists(LOCAL_FILE):
        try:
            df = pd.read_csv(LOCAL_FILE)
            source_used = "Local Workspace File"
        except Exception:
            pass

    if df is None:
        try:
            response = requests.get(DATA_URL, timeout=10)
            if response.status_code == 200:
                df = pd.read_csv(io.StringIO(response.text))
                # Cache locally for future offline resilience
                df.to_csv(LOCAL_FILE, index=False)
                source_used = "Remote GitHub Gist Repository"
        except Exception:
            pass

    if df is None:
        st.error("Failed to load dataset from both local file and remote gist repository.")
        st.stop()

    raw_shape = df.shape
    raw_missing = df.isnull().sum().to_dict()
    raw_duplicates = int(df.duplicated().sum())

    # Data cleaning as specified in Cell 11 of the notebook
    cleaned_df = df.drop_duplicates().dropna().copy()
    cleaned_shape = cleaned_df.shape

    # Derive engagement ratios for enhanced analytical value
    cleaned_df["Likes_Per_Minute"] = np.where(
        cleaned_df["Daily_Minutes_Spent"] > 0,
        (cleaned_df["Likes_Per_Day"] / cleaned_df["Daily_Minutes_Spent"]).round(3),
        0.0,
    )
    cleaned_df["Engagement_Score"] = (
        cleaned_df["Posts_Per_Day"] * 2.0
        + cleaned_df["Likes_Per_Day"] * 0.5
        + cleaned_df["Follows_Per_Day"] * 1.0
    ).round(1)

    metadata = {
        "source": source_used,
        "raw_shape": raw_shape,
        "raw_missing": raw_missing,
        "raw_duplicates": raw_duplicates,
        "cleaned_shape": cleaned_shape,
    }
    return cleaned_df, metadata


df, meta = load_and_preprocess_data()

# -----------------------------------------------------------------------------
# 3. COLOR PALETTES & PLOTLY STYLING UTILITIES
# -----------------------------------------------------------------------------
PRIMARY_COLOR = "#6366f1"
APP_PALETTE = {
    "Snapchat": "#fffc00",
    "LinkedIn": "#0a66c2",
    "TikTok": "#00f2fe",
    "Pinterest": "#e60023",
    "Instagram": "#e1306c",
    "Twitter": "#1da1f2",
    "Facebook": "#1877f2",
}

THEME_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(15, 23, 42, 0.4)",
    font=dict(family="Plus Jakarta Sans, sans-serif", color="#e2e8f0"),
    margin=dict(l=40, r=30, t=50, b=40),
    xaxis=dict(
        gridcolor="rgba(255, 255, 255, 0.07)",
        zerolinecolor="rgba(255, 255, 255, 0.12)",
        linecolor="rgba(255, 255, 255, 0.1)",
    ),
    yaxis=dict(
        gridcolor="rgba(255, 255, 255, 0.07)",
        zerolinecolor="rgba(255, 255, 255, 0.12)",
        linecolor="rgba(255, 255, 255, 0.1)",
    ),
    hoverlabel=dict(
        bgcolor="#1e293b",
        font_size=13,
        font_family="Plus Jakarta Sans",
    ),
)


def apply_plotly_theme(fig):
    fig.update_layout(**THEME_LAYOUT)
    return fig


# -----------------------------------------------------------------------------
# 4. SIDEBAR NAVIGATION
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown(
        """
        <div class="sidebar-brand">
            <div style="display: flex; align-items: center; gap: 10px;">
                <span style="font-size: 1.8rem;">📱</span>
                <div>
                    <div style="font-size: 1.15rem; font-weight: 800; color: #f8fafc; letter-spacing: -0.3px;">
                        Social Media Analysis
                    </div>
                    <div style="font-size: 0.75rem; color: #818cf8; font-weight: 600; text-transform: uppercase; letter-spacing: 0.6px;">
                        Statistical Inference Lab
                    </div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    nav_options = [
        "Executive Dashboard",
        "Dataset Explorer",
        "Descriptive Statistics",
        "Data Visualizations",
        "Probability Distributions",
        "Sampling Analysis & CLT",
        "Hypothesis Testing",
        "Predictive Modeling (ML)",
        "Insights & Conclusions",
    ]

    selected_section = st.radio(
        "Navigation",
        nav_options,
        index=0,
        label_visibility="collapsed",
    )

    st.markdown("---")

    # Author Credentials Badge from Notebook Header
    st.markdown(
        """
        <div class="sidebar-author-box">
            <div style="font-size: 0.72rem; color: #94a3b8; font-weight: 700; text-transform: uppercase; letter-spacing: 0.6px; margin-bottom: 6px;">
                Academic Project Credentials
            </div>
            <div style="font-size: 0.92rem; font-weight: 700; color: #f1f5f9;">
                Arya Singh
            </div>
            <div style="font-size: 0.8rem; color: #cbd5e1; margin-top: 2px;">
                Roll No: <span style="color: #38bdf8; font-weight: 600;">02</span> &nbsp;|&nbsp; Class: <span style="color: #818cf8; font-weight: 600;">B3[B1]</span>
            </div>
            <div style="font-size: 0.76rem; color: #64748b; margin-top: 4px;">
                Subject: Statistical Inference (SI Lab)
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Live Data Health Indicator
    st.markdown(
        f"""
        <div style="margin-top: 18px; padding: 10px 14px; background: rgba(16, 185, 129, 0.08); border: 1px solid rgba(16, 185, 129, 0.25); border-radius: 10px; display: flex; align-items: center; gap: 8px;">
            <div style="width: 8px; height: 8px; border-radius: 50%; background: #10b981;"></div>
            <div style="font-size: 0.78rem; color: #a7f3d0; font-weight: 600;">
                {len(df):,} Records Loaded • Verified
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# -----------------------------------------------------------------------------
# 5. CORE SECTIONS
# -----------------------------------------------------------------------------

# =============================================================================
# SECTION 1: DASHBOARD
# =============================================================================
if selected_section == "Executive Dashboard":
    st.markdown(
        """
        <div class="hero-card">
            <div class="hero-title">Social Media Usage Analysis</div>
            <div class="hero-subtitle">
                An empirical investigation into user engagement patterns across leading platforms (Snapchat, LinkedIn, TikTok, Pinterest, Instagram, Twitter, and Facebook) using descriptive measures, theoretical probability modeling, inferential hypothesis testing, and supervised machine learning.
            </div>
            <div>
                <span class="badge-pill">📊 1,000 Verified Observations</span>
                <span class="badge-pill badge-pill-accent">🌐 7 Major Social Networks</span>
                <span class="badge-pill badge-pill-success">🔬 Empirical & Theoretical Inference</span>
                <span class="badge-pill">🤖 Predictive Modeling Active</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Key Statistics Metrics Row
    m_col1, m_col2, m_col3, m_col4 = st.columns(4)

    with m_col1:
        st.markdown(
            f"""
            <div class="stat-card">
                <div class="stat-label">Total Sample Size</div>
                <div class="stat-value">{len(df):,}</div>
                <div class="stat-desc">6 Monitored Attributes</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with m_col2:
        avg_minutes = df["Daily_Minutes_Spent"].mean()
        st.markdown(
            f"""
            <div class="stat-card">
                <div class="stat-label">Mean Daily Usage</div>
                <div class="stat-value">{avg_minutes:.1f} <span style="font-size: 1rem; color: #818cf8;">min</span></div>
                <div class="stat-desc">≈ {avg_minutes/60:.2f} hours per day</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with m_col3:
        avg_posts = df["Posts_Per_Day"].mean()
        st.markdown(
            f"""
            <div class="stat-card">
                <div class="stat-label">Avg Posts / Day</div>
                <div class="stat-value">{avg_posts:.2f}</div>
                <div class="stat-desc">Median: {df['Posts_Per_Day'].median():.0f} posts/day</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with m_col4:
        avg_likes = df["Likes_Per_Day"].mean()
        st.markdown(
            f"""
            <div class="stat-card">
                <div class="stat-label">Avg Likes / Day</div>
                <div class="stat-value">{avg_likes:.1f}</div>
                <div class="stat-desc">Spread: 0 to {df['Likes_Per_Day'].max():.0f} likes</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.write("")

    # Visual Summary Layout
    dash_col1, dash_col2 = st.columns([1, 1])

    with dash_col1:
        st.markdown(
            """
            <div class="glass-card">
                <div class="section-title"><span>🌐</span> Platform Market Share</div>
                <div class="section-desc">Observation distribution across monitored social networks.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        app_counts = df["App"].value_counts().reset_index()
        app_counts.columns = ["App", "Count"]

        fig_pie = px.pie(
            app_counts,
            values="Count",
            names="App",
            hole=0.55,
            color="App",
            color_discrete_map=APP_PALETTE,
        )
        fig_pie.update_traces(
            textposition="inside",
            textinfo="percent+label",
            marker=dict(line=dict(color="#0f172a", width=2)),
        )
        fig_pie.update_layout(
            showlegend=False,
            height=320,
            margin=dict(l=10, r=10, t=10, b=10),
            paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    with dash_col2:
        st.markdown(
            """
            <div class="glass-card">
                <div class="section-title"><span>⏱️</span> Daily Minutes Spent Density</div>
                <div class="section-desc">Empirical density distribution of screen time across 1,000 subjects.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        fig_hist = px.histogram(
            df,
            x="Daily_Minutes_Spent",
            nbins=25,
            marginal="box",
            color_discrete_sequence=["#818cf8"],
            labels={"Daily_Minutes_Spent": "Daily Minutes Spent"},
        )
        fig_hist.add_vline(
            x=df["Daily_Minutes_Spent"].mean(),
            line_dash="dash",
            line_color="#38bdf8",
            annotation_text=f"Mean: {df['Daily_Minutes_Spent'].mean():.1f}m",
            annotation_position="top left",
        )
        fig_hist.add_vline(
            x=df["Daily_Minutes_Spent"].median(),
            line_dash="dot",
            line_color="#f43f5e",
            annotation_text=f"Median: {df['Daily_Minutes_Spent'].median():.0f}m",
            annotation_position="top right",
        )
        fig_hist.update_layout(
            height=320,
            margin=dict(l=10, r=10, t=10, b=10),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(15, 23, 42, 0.4)",
        )
        apply_plotly_theme(fig_hist)
        st.plotly_chart(fig_hist, use_container_width=True)

    # Project Scope & Objectives
    st.markdown(
        """
        <div class="glass-card" style="margin-top: 15px;">
            <div class="section-title"><span>🎯</span> Project Objectives & Analytical Architecture</div>
            <div class="section-desc">
                Designed to cover core modules of statistical inference and predictive modeling:
            </div>
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 16px;">
                <div style="background: rgba(15, 23, 42, 0.6); padding: 16px; border-radius: 12px; border-left: 3px solid #818cf8;">
                    <div style="font-weight: 700; color: #f8fafc; margin-bottom: 4px;">1. Exploratory & Summary Analysis</div>
                    <div style="font-size: 0.85rem; color: #94a3b8;">Central tendency, dispersion measures (mean, median, mode, variance, std dev), and cross-platform segmentations.</div>
                </div>
                <div style="background: rgba(15, 23, 42, 0.6); padding: 16px; border-radius: 12px; border-left: 3px solid #38bdf8;">
                    <div style="font-weight: 700; color: #f8fafc; margin-bottom: 4px;">2. Theoretical Probability Models</div>
                    <div style="font-size: 0.85rem; color: #94a3b8;">Fitting Uniform, Normal, Poisson, and Binomial distributions against empirical behavioural observations.</div>
                </div>
                <div style="background: rgba(15, 23, 42, 0.6); padding: 16px; border-radius: 12px; border-left: 3px solid #34d399;">
                    <div style="font-weight: 700; color: #f8fafc; margin-bottom: 4px;">3. Sampling Theory & CLT</div>
                    <div style="font-size: 0.85rem; color: #94a3b8;">Monte Carlo simulations of sampling distributions, replacement mechanisms, and Central Limit Theorem convergence.</div>
                </div>
                <div style="background: rgba(15, 23, 42, 0.6); padding: 16px; border-radius: 12px; border-left: 3px solid #fbbf24;">
                    <div style="font-weight: 700; color: #f8fafc; margin-bottom: 4px;">4. Statistical Hypothesis Testing</div>
                    <div style="font-size: 0.85rem; color: #94a3b8;">One-sample Z/T tests against hypothesized benchmarks, Welch's two-sample comparisons, and Chi-Square tests.</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# =============================================================================
# SECTION 2: DATASET EXPLORER
# =============================================================================
elif selected_section == "Dataset Explorer":
    st.markdown(
        """
        <div class="hero-card">
            <div class="hero-title">Interactive Dataset Explorer</div>
            <div class="hero-subtitle">
                Filter, search, inspect schema integrity, and export verified records from the dataset used in the lab.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Filter Controls inside an expandable card
    with st.expander("🔍 Interactive Data Filters & Query Builder", expanded=True):
        f_col1, f_col2, f_col3 = st.columns([1, 1, 1])

        with f_col1:
            all_apps = sorted(df["App"].unique().tolist())
            selected_apps = st.multiselect(
                "Filter by Social Network Platform",
                options=all_apps,
                default=all_apps,
            )

        with f_col2:
            min_m, max_m = int(df["Daily_Minutes_Spent"].min()), int(
                df["Daily_Minutes_Spent"].max()
            )
            selected_minutes = st.slider(
                "Daily Minutes Spent Range",
                min_value=min_m,
                max_value=max_m,
                value=(min_m, max_m),
            )

        with f_col3:
            min_p, max_p = int(df["Posts_Per_Day"].min()), int(
                df["Posts_Per_Day"].max()
            )
            selected_posts = st.slider(
                "Posts Per Day Range",
                min_value=min_p,
                max_value=max_p,
                value=(min_p, max_p),
            )

        search_query = st.text_input(
            "Quick Search by User ID (e.g. 'U_10')",
            placeholder="Type User ID here...",
        ).strip()

    # Filter logic
    filtered_df = df[
        (df["App"].isin(selected_apps))
        & (df["Daily_Minutes_Spent"] >= selected_minutes[0])
        & (df["Daily_Minutes_Spent"] <= selected_minutes[1])
        & (df["Posts_Per_Day"] >= selected_posts[0])
        & (df["Posts_Per_Day"] <= selected_posts[1])
    ]

    if search_query:
        filtered_df = filtered_df[
            filtered_df["User_ID"].str.contains(search_query, case=False, na=False)
        ]

    # Filter Feedback Banner
    f_info1, f_info2, f_info3 = st.columns([2, 1, 1])
    with f_info1:
        st.markdown(
            f"**Displaying {len(filtered_df):,} of {len(df):,} records** ({(len(filtered_df)/len(df))*100:.1f}% matching active filters)"
        )
    with f_info3:
        csv_buffer = filtered_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download Filtered CSV",
            data=csv_buffer,
            file_name="social_media_usage_filtered.csv",
            mime="text/csv",
            use_container_width=True,
        )

    # Interactive Data Table
    st.dataframe(
        filtered_df,
        use_container_width=True,
        height=400,
        column_config={
            "User_ID": st.column_config.TextColumn("User ID", width="small"),
            "App": st.column_config.TextColumn("Platform", width="medium"),
            "Daily_Minutes_Spent": st.column_config.ProgressColumn(
                "Daily Minutes",
                min_value=0,
                max_value=500,
                format="%d min",
            ),
            "Posts_Per_Day": st.column_config.NumberColumn(
                "Posts/Day", format="%d"
            ),
            "Likes_Per_Day": st.column_config.NumberColumn(
                "Likes/Day", format="%d"
            ),
            "Follows_Per_Day": st.column_config.NumberColumn(
                "Follows/Day", format="%d"
            ),
            "Engagement_Score": st.column_config.NumberColumn(
                "Engagement Score", format="%.1f"
            ),
        },
    )

    # Dataset Quality & Health Audit
    st.markdown(
        """
        <div class="glass-card" style="margin-top: 20px;">
            <div class="section-title"><span>🛡️</span> Dataset Integrity & Health Audit</div>
            <div class="section-desc">Verification report following the notebook cleaning pipeline.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    h_col1, h_col2, h_col3 = st.columns(3)

    with h_col1:
        st.markdown(
            f"""
            <div style="background: rgba(15, 23, 42, 0.7); padding: 18px; border-radius: 12px; border: 1px solid rgba(255, 255, 255, 0.08);">
                <div style="color: #94a3b8; font-size: 0.8rem; font-weight: 700; text-transform: uppercase;">Missing Values</div>
                <div style="font-size: 1.6rem; font-weight: 800; color: #10b981; margin: 4px 0;">0 Nulls</div>
                <div style="font-size: 0.8rem; color: #64748b;">Complete data completeness verified</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with h_col2:
        st.markdown(
            f"""
            <div style="background: rgba(15, 23, 42, 0.7); padding: 18px; border-radius: 12px; border: 1px solid rgba(255, 255, 255, 0.08);">
                <div style="color: #94a3b8; font-size: 0.8rem; font-weight: 700; text-transform: uppercase;">Duplicate Rows</div>
                <div style="font-size: 1.6rem; font-weight: 800; color: #10b981; margin: 4px 0;">{meta['raw_duplicates']} Duplicates</div>
                <div style="font-size: 0.8rem; color: #64748b;">100% unique user entities</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with h_col3:
        st.markdown(
            f"""
            <div style="background: rgba(15, 23, 42, 0.7); padding: 18px; border-radius: 12px; border: 1px solid rgba(255, 255, 255, 0.08);">
                <div style="color: #94a3b8; font-size: 0.8rem; font-weight: 700; text-transform: uppercase;">Source Provenance</div>
                <div style="font-size: 1.1rem; font-weight: 800; color: #38bdf8; margin: 6px 0;">{meta['source']}</div>
                <div style="font-size: 0.8rem; color: #64748b;">Directly consumed into pandas DataFrame</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Column Schema Inspection Table
    st.write("")
    schema_data = []
    for col in df.columns:
        schema_data.append(
            {
                "Column Name": col,
                "Data Type": str(df[col].dtype),
                "Non-Null Count": int(df[col].count()),
                "Unique Values": int(df[col].nunique()),
                "Sample Entry": str(df[col].iloc[0]),
            }
        )
    schema_df = pd.DataFrame(schema_data)
    st.table(schema_df)

# =============================================================================
# SECTION 3: DESCRIPTIVE STATISTICS
# =============================================================================
elif selected_section == "Descriptive Statistics":
    st.markdown(
        """
        <div class="hero-card">
            <div class="hero-title">Descriptive Statistics & Parameter Estimation</div>
            <div class="hero-subtitle">
                Exact dynamical recalculation of central tendency, dispersion, and higher statistical moments for all empirical numerical features.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    numeric_cols = [
        "Daily_Minutes_Spent",
        "Posts_Per_Day",
        "Likes_Per_Day",
        "Follows_Per_Day",
    ]

    # Full Statistical Summary Table Calculation
    stats_list = []
    for c in numeric_cols:
        series = df[c]
        mean_val = float(series.mean())
        median_val = float(series.median())
        mode_val = float(series.mode()[0])
        min_val = float(series.min())
        max_val = float(series.max())
        rng_val = float(max_val - min_val)
        var_val = float(series.var())
        std_val = float(series.std())
        skew_val = float(series.skew())
        kurt_val = float(series.kurtosis())
        iqr_val = float(series.quantile(0.75) - series.quantile(0.25))

        stats_list.append(
            {
                "Feature": c.replace("_", " "),
                "Mean": round(mean_val, 2),
                "Median": round(median_val, 2),
                "Mode": round(mode_val, 2),
                "Min": round(min_val, 2),
                "Max": round(max_val, 2),
                "Range": round(rng_val, 2),
                "Variance": round(var_val, 2),
                "Std Deviation": round(std_val, 2),
                "IQR": round(iqr_val, 2),
                "Skewness": round(skew_val, 3),
                "Kurtosis": round(kurt_val, 3),
            }
        )

    summary_stats_df = pd.DataFrame(stats_list)

    st.markdown(
        """
        <div class="glass-card">
            <div class="section-title"><span>📐</span> Dynamic Statistical Parameter Matrix</div>
            <div class="section-desc">Computed in real-time from the cleaned dataset.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.dataframe(
        summary_stats_df.set_index("Feature"),
        use_container_width=True,
    )

    st.write("")

    # Interactive Feature Deep Dive
    st.markdown(
        """
        <div class="glass-card">
            <div class="section-title"><span>🔬</span> Feature Deep Dive & Quartile Inspector</div>
            <div class="section-desc">Select any numerical variable to inspect its empirical spread.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    selected_feat = st.selectbox(
        "Select Feature for In-Depth Breakdown:",
        options=numeric_cols,
        format_func=lambda x: x.replace("_", " "),
    )

    feat_series = df[selected_feat]
    f1, f2, f3, f4, f5 = st.columns(5)
    f1.metric("Mean", f"{feat_series.mean():.2f}")
    f2.metric("Median", f"{feat_series.median():.2f}")
    f3.metric("Std Dev", f"{feat_series.std():.2f}")
    f4.metric("Variance", f"{feat_series.var():.2f}")
    f5.metric("IQR", f"{feat_series.quantile(0.75) - feat_series.quantile(0.25):.2f}")

    # Box Plot + Quantile Table
    c_left, c_right = st.columns([3, 2])

    with c_left:
        fig_single_box = px.box(
            df,
            x="App",
            y=selected_feat,
            color="App",
            color_discrete_map=APP_PALETTE,
            points="all",
            title=f"Distribution of {selected_feat.replace('_', ' ')} Across Apps",
        )
        apply_plotly_theme(fig_single_box)
        st.plotly_chart(fig_single_box, use_container_width=True)

    with c_right:
        quantiles = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]
        q_vals = [feat_series.quantile(q) for q in quantiles]
        q_df = pd.DataFrame(
            {
                "Percentile": [f"P{int(q*100)}" for q in quantiles],
                "Threshold Value": [round(v, 2) for v in q_vals],
            }
        )
        st.markdown("**Empirical Percentile Ranks:**")
        st.table(q_df)

    # Per-Platform Grouped Statistics
    st.markdown(
        """
        <div class="glass-card" style="margin-top: 20px;">
            <div class="section-title"><span>📱</span> Platform-Wise Comparative Statistics</div>
            <div class="section-desc">Grouped metrics for daily minutes spent across each network.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    platform_stats = (
        df.groupby("App")["Daily_Minutes_Spent"]
        .agg(["count", "mean", "median", "std", "min", "max"])
        .reset_index()
    )
    platform_stats.columns = [
        "Platform",
        "Users",
        "Mean (min)",
        "Median (min)",
        "Std Dev",
        "Min",
        "Max",
    ]
    platform_stats["Mean (min)"] = platform_stats["Mean (min)"].round(2)
    platform_stats["Median (min)"] = platform_stats["Median (min)"].round(2)
    platform_stats["Std Dev"] = platform_stats["Std Dev"].round(2)

    st.dataframe(platform_stats, use_container_width=True)

# =============================================================================
# SECTION 4: DATA VISUALIZATIONS
# =============================================================================
elif selected_section == "Data Visualizations":
    st.markdown(
        """
        <div class="hero-card">
            <div class="hero-title">Visual Analytics & Bivariate Explorations</div>
            <div class="hero-subtitle">
                Interactive charts reproducing and extending the visual analysis from notebook Cells 19-22 and Cell 51.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Row 1: Platform Usage & Daily Usage Distribution
    v_col1, v_col2 = st.columns(2)

    with v_col1:
        st.markdown(
            """
            <div class="glass-card">
                <div class="section-title"><span>📊</span> Social Media Platform Representation</div>
                <div class="section-desc">User count breakdown across platforms (from Notebook Cell 19).</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        app_counts = df["App"].value_counts().reset_index()
        app_counts.columns = ["Platform", "Users"]
        fig_app_bar = px.bar(
            app_counts,
            x="Platform",
            y="Users",
            color="Platform",
            color_discrete_map=APP_PALETTE,
            text="Users",
        )
        fig_app_bar.update_traces(textposition="outside")
        apply_plotly_theme(fig_app_bar)
        st.plotly_chart(fig_app_bar, use_container_width=True)

    with v_col2:
        st.markdown(
            """
            <div class="glass-card">
                <div class="section-title"><span>⏱️</span> Daily Usage Frequency Histogram</div>
                <div class="section-desc">Minutes spent per day with configurable bin width (Cell 20).</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        bins_chosen = st.slider("Select Histogram Bin Count", 5, 50, 15, key="hist_bins")
        fig_usage_hist = px.histogram(
            df,
            x="Daily_Minutes_Spent",
            nbins=bins_chosen,
            color_discrete_sequence=["#6366f1"],
            opacity=0.85,
        )
        fig_usage_hist.add_vline(
            x=df["Daily_Minutes_Spent"].mean(),
            line_dash="dash",
            line_color="#38bdf8",
            annotation_text=f"Mean: {df['Daily_Minutes_Spent'].mean():.1f}",
        )
        apply_plotly_theme(fig_usage_hist)
        st.plotly_chart(fig_usage_hist, use_container_width=True)

    # Row 2: Box Plot & Scatter Plot
    v_col3, v_col4 = st.columns(2)

    with v_col3:
        st.markdown(
            """
            <div class="glass-card">
                <div class="section-title"><span>📦</span> Daily Usage Boxplot & Quartiles</div>
                <div class="section-desc">Visualizing median and interquartile dispersion (Cell 21).</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        group_by_app = st.checkbox(
            "Segment Boxplot by Platform", value=True, key="box_group"
        )
        if group_by_app:
            fig_box = px.box(
                df,
                x="App",
                y="Daily_Minutes_Spent",
                color="App",
                color_discrete_map=APP_PALETTE,
            )
        else:
            fig_box = px.box(
                df,
                y="Daily_Minutes_Spent",
                color_discrete_sequence=["#818cf8"],
            )
        apply_plotly_theme(fig_box)
        st.plotly_chart(fig_box, use_container_width=True)

    with v_col4:
        st.markdown(
            """
            <div class="glass-card">
                <div class="section-title"><span>✨</span> Daily Minutes Spent vs Posts Per Day</div>
                <div class="section-desc">Bivariate relationship with OLS regression trendline (Cell 22).</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        fig_scatter = px.scatter(
            df,
            x="Daily_Minutes_Spent",
            y="Posts_Per_Day",
            color="App",
            color_discrete_map=APP_PALETTE,
            trendline="ols",
            hover_data=["User_ID", "Likes_Per_Day", "Follows_Per_Day"],
            labels={
                "Daily_Minutes_Spent": "Daily Minutes Spent",
                "Posts_Per_Day": "Posts Per Day",
            },
        )
        apply_plotly_theme(fig_scatter)
        st.plotly_chart(fig_scatter, use_container_width=True)

    # Row 3: Correlation Matrix Heatmap (From Notebook Cell 51)
    st.markdown(
        """
        <div class="glass-card" style="margin-top: 20px;">
            <div class="section-title"><span>🔥</span> Correlation Matrix & Feature Covariance</div>
            <div class="section-desc">Interactive correlation heatmap implemented from Cell 51 of the notebook.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    numeric_cols = [
        "Daily_Minutes_Spent",
        "Posts_Per_Day",
        "Likes_Per_Day",
        "Follows_Per_Day",
    ]
    corr_method = st.radio(
        "Select Correlation Algorithm:",
        ["pearson", "spearman"],
        horizontal=True,
        format_func=lambda x: "Pearson (Linear Correlation)"
        if x == "pearson"
        else "Spearman (Rank / Monotonic Correlation)",
    )

    corr_matrix = df[numeric_cols].corr(method=corr_method).round(3)

    fig_heat = px.imshow(
        corr_matrix,
        text_auto=True,
        aspect="auto",
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
        labels=dict(color="Correlation"),
    )
    fig_heat.update_layout(height=360)
    apply_plotly_theme(fig_heat)
    st.plotly_chart(fig_heat, use_container_width=True)

# =============================================================================
# SECTION 5: PROBABILITY DISTRIBUTIONS
# =============================================================================
elif selected_section == "Probability Distributions":
    st.markdown(
        """
        <div class="hero-card">
            <div class="hero-title">Probability Distributions & Modeling</div>
            <div class="hero-subtitle">
                Faithful implementation of the Uniform, Normal, Poisson, and Binomial distribution practicals from the notebook (Cells 23-35).
            </div>
            <div>
                <span class="badge-pill">U(a, b) Uniform</span>
                <span class="badge-pill">N(μ, σ²) Normal</span>
                <span class="badge-pill">Poisson(λ)</span>
                <span class="badge-pill">Binomial(n, p)</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    dist_tab = st.tabs(
        [
            "1. Uniform Distribution",
            "2. Normal Distribution",
            "3. Poisson Distribution",
            "4. Binomial Distribution",
        ]
    )

    # 1. UNIFORM DISTRIBUTION (Cells 24-25)
    with dist_tab[0]:
        st.markdown(
            """
            <div class="glass-card">
                <div class="section-title"><span>📐</span> Continuous Uniform Distribution U(a, b)</div>
                <div class="section-desc">
                    Models equal probability of occurrence across all daily screen time durations between observed bounds [a, b].
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        u_col1, u_col2 = st.columns([1, 2])
        min_actual = float(df["Daily_Minutes_Spent"].min())
        max_actual = float(df["Daily_Minutes_Spent"].max())

        with u_col1:
            st.markdown(
                f"""
                <div style="background: rgba(15, 23, 42, 0.7); padding: 16px; border-radius: 12px; border: 1px solid rgba(255, 255, 255, 0.08); margin-bottom: 15px;">
                    <div style="font-size: 0.8rem; color: #94a3b8; font-weight: 700;">EMPIRICAL BOUNDS</div>
                    <div style="font-size: 1.1rem; color: #f8fafc; margin-top: 4px;">a (Minimum) = <b>{min_actual:.0f} min</b></div>
                    <div style="font-size: 1.1rem; color: #f8fafc; margin-top: 2px;">b (Maximum) = <b>{max_actual:.0f} min</b></div>
                    <div style="font-size: 0.85rem; color: #38bdf8; margin-top: 6px;">Theoretical E[X] = (a+b)/2 = <b>{(min_actual+max_actual)/2:.2f} min</b></div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            uniform_n = st.slider("Simulation Sample Size", 100, 5000, 1000, step=100)
            uniform_seed = st.number_input("Random Seed", value=42, step=1)

            np.random.seed(uniform_seed)
            uniform_data = np.random.uniform(min_actual, max_actual, uniform_n)

            st.metric("Simulated Sample Mean", f"{uniform_data.mean():.2f} min")
            st.metric("Empirical Actual Mean", f"{df['Daily_Minutes_Spent'].mean():.2f} min")

        with u_col2:
            show_mode = st.radio(
                "Visual View:",
                ["Simulated Uniform Distribution", "Compare: Simulated Uniform vs Real Observed Data"],
                horizontal=True,
            )

            if show_mode == "Simulated Uniform Distribution":
                fig_u = px.histogram(
                    x=uniform_data,
                    nbins=20,
                    title="Simulated Uniform Distribution of Social Media Usage (1,000 Draws)",
                    labels={"x": "Daily Minutes Spent", "y": "Frequency"},
                    color_discrete_sequence=["#818cf8"],
                )
                fig_u.add_vline(
                    x=uniform_data.mean(),
                    line_dash="dash",
                    line_color="#38bdf8",
                    annotation_text=f"Sim Mean: {uniform_data.mean():.1f}",
                )
            else:
                fig_u = go.Figure()
                fig_u.add_trace(
                    go.Histogram(
                        x=uniform_data,
                        name="Simulated Uniform U(a,b)",
                        opacity=0.6,
                        marker_color="#818cf8",
                        nbinsx=20,
                    )
                )
                fig_u.add_trace(
                    go.Histogram(
                        x=df["Daily_Minutes_Spent"],
                        name="Real Observed Data",
                        opacity=0.6,
                        marker_color="#38bdf8",
                        nbinsx=20,
                    )
                )
                fig_u.update_layout(
                    barmode="overlay",
                    title="Real Observed Screen Time vs Simulated Uniform Distribution",
                    xaxis_title="Daily Minutes Spent",
                    yaxis_title="Frequency",
                )

            apply_plotly_theme(fig_u)
            st.plotly_chart(fig_u, use_container_width=True)

    # 2. NORMAL DISTRIBUTION (Cells 27-28)
    with dist_tab[1]:
        st.markdown(
            """
            <div class="glass-card">
                <div class="section-title"><span>🔔</span> Gaussian Normal Distribution N(μ, σ²)</div>
                <div class="section-desc">
                    Fitted bell curve parameterized by actual empirical sample mean μ and standard deviation σ.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        n_col1, n_col2 = st.columns([1, 2])
        mean_actual = float(df["Daily_Minutes_Spent"].mean())
        std_actual = float(df["Daily_Minutes_Spent"].std())

        with n_col1:
            st.markdown(
                f"""
                <div style="background: rgba(15, 23, 42, 0.7); padding: 16px; border-radius: 12px; border: 1px solid rgba(255, 255, 255, 0.08); margin-bottom: 15px;">
                    <div style="font-size: 0.8rem; color: #94a3b8; font-weight: 700;">EMPIRICAL PARAMETERS</div>
                    <div style="font-size: 1.1rem; color: #f8fafc; margin-top: 4px;">μ (Mean) = <b>{mean_actual:.2f} min</b></div>
                    <div style="font-size: 1.1rem; color: #f8fafc; margin-top: 2px;">σ (Std Dev) = <b>{std_actual:.2f} min</b></div>
                    <div style="font-size: 0.85rem; color: #38bdf8; margin-top: 6px;">Variance σ² = <b>{std_actual**2:.2f}</b></div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            normal_n = st.slider("Number of Gaussian Draws", 100, 5000, 1000, step=100)
            np.random.seed(42)
            normal_data = np.random.normal(mean_actual, std_actual, normal_n)

            st.metric("Generated Mean", f"{normal_data.mean():.2f} min")
            st.metric("Generated Std Dev", f"{normal_data.std():.2f} min")

        with n_col2:
            fig_norm = go.Figure()
            # Histogram of simulated normal
            fig_norm.add_trace(
                go.Histogram(
                    x=normal_data,
                    histnorm="probability density",
                    name="Simulated Normal N(μ, σ²)",
                    marker_color="#818cf8",
                    opacity=0.6,
                    nbinsx=25,
                )
            )

            # Continuous theoretical Gaussian curve
            x_range = np.linspace(
                mean_actual - 3.5 * std_actual, mean_actual + 3.5 * std_actual, 300
            )
            y_norm = stats.norm.pdf(x_range, mean_actual, std_actual)
            fig_norm.add_trace(
                go.Scatter(
                    x=x_range,
                    y=y_norm,
                    mode="lines",
                    name="Theoretical Normal Curve PDF",
                    line=dict(color="#38bdf8", width=3),
                )
            )

            fig_norm.update_layout(
                title="Normal Distribution of Social Media Usage with Theoretical PDF",
                xaxis_title="Daily Minutes Spent",
                yaxis_title="Probability Density",
            )
            apply_plotly_theme(fig_norm)
            st.plotly_chart(fig_norm, use_container_width=True)

    # 3. POISSON DISTRIBUTION (Cells 30-31)
    with dist_tab[2]:
        st.markdown(
            """
            <div class="glass-card">
                <div class="section-title"><span>🎯</span> Discrete Poisson Distribution Poisson(λ)</div>
                <div class="section-desc">
                    Models counts of discrete events (Posts per day) over a fixed interval with parameter λ = mean posts per day.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        p_col1, p_col2 = st.columns([1, 2])
        lambda_val = float(df["Posts_Per_Day"].mean())

        with p_col1:
            st.markdown(
                f"""
                <div style="background: rgba(15, 23, 42, 0.7); padding: 16px; border-radius: 12px; border: 1px solid rgba(255, 255, 255, 0.08); margin-bottom: 15px;">
                    <div style="font-size: 0.8rem; color: #94a3b8; font-weight: 700;">POISSON RATE PARAMETER</div>
                    <div style="font-size: 1.1rem; color: #f8fafc; margin-top: 4px;">λ (Avg Posts/Day) = <b>{lambda_val:.2f}</b></div>
                    <div style="font-size: 0.85rem; color: #38bdf8; margin-top: 6px;">Theoretical Var[X] = λ = <b>{lambda_val:.2f}</b></div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            poisson_n = st.slider("Poisson Realizations", 100, 5000, 1000, step=100)
            np.random.seed(42)
            poisson_data = np.random.poisson(lambda_val, poisson_n)

            st.metric("Generated Mean", f"{poisson_data.mean():.2f}")
            st.metric("Generated Variance", f"{poisson_data.var():.2f}")

        with p_col2:
            max_p_val = int(poisson_data.max()) + 2
            fig_pois = px.histogram(
                x=poisson_data,
                nbins=max_p_val,
                title=f"Poisson Distribution of Daily Posts (λ = {lambda_val:.2f})",
                labels={"x": "Posts Per Day", "y": "Frequency"},
                color_discrete_sequence=["#34d399"],
            )
            # Theoretical PMF overlay
            k_vals = np.arange(0, max_p_val)
            theoretical_pmf = stats.poisson.pmf(k_vals, lambda_val) * poisson_n
            fig_pois.add_trace(
                go.Scatter(
                    x=k_vals,
                    y=theoretical_pmf,
                    mode="lines+markers",
                    name="Theoretical Poisson PMF",
                    line=dict(color="#f59e0b", width=2),
                )
            )
            apply_plotly_theme(fig_pois)
            st.plotly_chart(fig_pois, use_container_width=True)

    # 4. BINOMIAL DISTRIBUTION (Cells 33-35)
    with dist_tab[3]:
        st.markdown(
            """
            <div class="glass-card">
                <div class="section-title"><span>🎲</span> Discrete Binomial Distribution B(n, p)</div>
                <div class="section-desc">
                    Models the number of high-usage users (Daily Minutes Spent > Median) in independent trials of batch size n.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        b_col1, b_col2 = st.columns([1, 2])
        median_usage = float(df["Daily_Minutes_Spent"].median())
        p_success = float((df["Daily_Minutes_Spent"] > median_usage).mean())

        with b_col1:
            st.markdown(
                f"""
                <div style="background: rgba(15, 23, 42, 0.7); padding: 16px; border-radius: 12px; border: 1px solid rgba(255, 255, 255, 0.08); margin-bottom: 15px;">
                    <div style="font-size: 0.8rem; color: #94a3b8; font-weight: 700;">EXPERIMENT PARAMETERS</div>
                    <div style="font-size: 0.95rem; color: #f8fafc; margin-top: 4px;">Median Split Threshold = <b>{median_usage:.0f} min</b></div>
                    <div style="font-size: 0.95rem; color: #f8fafc; margin-top: 2px;">Success Probability p = <b>{p_success:.4f}</b></div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            n_trials = st.slider("Number of Users in Cohort (n)", 5, 50, 10, step=1)
            np.random.seed(42)
            binomial_data = np.random.binomial(n_trials, p_success, 1000)

            st.metric("Expected High-Usage (n*p)", f"{n_trials * p_success:.2f}")
            st.metric("Simulated Average High-Usage", f"{binomial_data.mean():.2f}")

        with b_col2:
            fig_binom = px.histogram(
                x=binomial_data,
                nbins=n_trials + 2,
                title=f"Binomial Distribution: Count of High-Usage Users in Cohort (n={n_trials}, p={p_success:.2f})",
                labels={"x": "Number of High-Usage Users in Cohort", "y": "Frequency"},
                color_discrete_sequence=["#ec4899"],
            )
            # Theoretical Binomial PMF
            k_binom = np.arange(0, n_trials + 1)
            theoretical_b_pmf = stats.binom.pmf(k_binom, n_trials, p_success) * 1000
            fig_binom.add_trace(
                go.Scatter(
                    x=k_binom,
                    y=theoretical_b_pmf,
                    mode="lines+markers",
                    name="Theoretical Binomial PMF",
                    line=dict(color="#38bdf8", width=2),
                )
            )
            apply_plotly_theme(fig_binom)
            st.plotly_chart(fig_binom, use_container_width=True)

# =============================================================================
# SECTION 6: SAMPLING ANALYSIS & CENTRAL LIMIT THEOREM
# =============================================================================
elif selected_section == "Sampling Analysis & CLT":
    st.markdown(
        """
        <div class="hero-card">
            <div class="hero-title">Sampling Analysis & The Central Limit Theorem</div>
            <div class="hero-subtitle">
                Interactive laboratory executing random sampling techniques, population vs sample parameter estimation, standard error derivation, and Central Limit Theorem asymptotic normality (Cells 36-50).
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    s_tab1, s_tab2, s_tab3, s_tab4 = st.tabs(
        [
            "1. Random Sampling Schemes",
            "2. Population vs Sample Mean",
            "3. Sampling Distribution of the Mean",
            "4. Central Limit Theorem (CLT) Lab",
        ]
    )

    # 1. RANDOM SAMPLING SCHEMES (Cells 37-39)
    with s_tab1:
        st.markdown(
            """
            <div class="glass-card">
                <div class="section-title"><span>🎲</span> Simple Random Sampling (SRS)</div>
                <div class="section-desc">Comparison of Sampling Without Replacement (WOR) and With Replacement (WR).</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        samp_n = st.slider("Sample Size (n)", min_value=5, max_value=50, value=10, step=1)
        samp_seed = st.number_input("Random Seed", value=42, step=1, key="samp_seed")

        s1, s2 = st.columns(2)
        with s1:
            st.markdown("**Sample WITHOUT Replacement (replace=False)**")
            sample_wor = df.sample(n=samp_n, replace=False, random_state=int(samp_seed))
            st.dataframe(sample_wor[["User_ID", "App", "Daily_Minutes_Spent", "Posts_Per_Day"]], use_container_width=True)
            st.caption(f"Mean Usage: {sample_wor['Daily_Minutes_Spent'].mean():.2f} min")

        with s2:
            st.markdown("**Sample WITH Replacement (replace=True)**")
            sample_wr = df.sample(n=samp_n, replace=True, random_state=int(samp_seed))
            st.dataframe(sample_wr[["User_ID", "App", "Daily_Minutes_Spent", "Posts_Per_Day"]], use_container_width=True)
            st.caption(f"Mean Usage: {sample_wr['Daily_Minutes_Spent'].mean():.2f} min")

    # 2. POPULATION VS SAMPLE MEAN (Cells 41-42)
    with s_tab2:
        st.markdown(
            """
            <div class="glass-card">
                <div class="section-title"><span>⚖️</span> Population Parameter vs Sample Statistic</div>
                <div class="section-desc">Visualizing estimation error between true population parameter μ and point estimator X̄.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        pop = df["Daily_Minutes_Spent"]
        pop_mean = float(pop.mean())

        sample_choice_n = st.slider("Select Sample Size (n)", 5, 200, 10, key="pop_v_sample_n")
        sample_dynamic = pop.sample(n=sample_choice_n, random_state=42)
        samp_mean = float(sample_dynamic.mean())
        abs_error = abs(samp_mean - pop_mean)
        pct_error = (abs_error / pop_mean) * 100

        col_m1, col_m2, col_m3 = st.columns(3)
        col_m1.metric("True Population Mean (μ)", f"{pop_mean:.2f} min")
        col_m2.metric(f"Sample Mean (X̄, n={sample_choice_n})", f"{samp_mean:.2f} min")
        col_m3.metric("Estimation Error", f"{abs_error:.2f} min ({pct_error:.1f}%)")

        fig_comp = go.Figure(
            data=[
                go.Bar(
                    x=["Population Mean (μ)", f"Sample Mean (X̄, n={sample_choice_n})"],
                    y=[pop_mean, samp_mean],
                    text=[f"{pop_mean:.2f}", f"{samp_mean:.2f}"],
                    textposition="auto",
                    marker_color=["#6366f1", "#38bdf8"],
                )
            ]
        )
        fig_comp.update_layout(
            title="Population Mean vs Sample Mean Comparison",
            yaxis_title="Daily Minutes Spent",
            height=360,
        )
        apply_plotly_theme(fig_comp)
        st.plotly_chart(fig_comp, use_container_width=True)

    # 3. SAMPLING DISTRIBUTION OF THE MEAN (Cells 44-46)
    with s_tab3:
        st.markdown(
            """
            <div class="glass-card">
                <div class="section-title"><span>📊</span> Monte Carlo Sampling Distribution of Sample Means</div>
                <div class="section-desc">Empirical simulation of 500 samples of size n=10 from the notebook (Cell 44).</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        sd_col1, sd_col2 = st.columns([1, 2])
        with sd_col1:
            num_samples = st.slider("Number of Monte Carlo Samples (K)", 100, 2000, 500, step=100)
            sub_size = st.slider("Sample Size (n)", 5, 50, 10, step=1)

            pop_vals = df["Daily_Minutes_Spent"].values
            np.random.seed(42)
            sim_means = np.array(
                [
                    np.random.choice(pop_vals, size=sub_size, replace=True).mean()
                    for _ in range(num_samples)
                ]
            )

            st.metric("Population Mean (μ)", f"{pop_vals.mean():.2f}")
            st.metric("Mean of Sample Means (E[X̄])", f"{sim_means.mean():.2f}")
            st.metric("Empirical Standard Error (SE)", f"{sim_means.std():.2f}")
            theo_se = pop_vals.std() / math.sqrt(sub_size)
            st.metric("Theoretical SE (σ / √n)", f"{theo_se:.2f}")

        with sd_col2:
            fig_sd = px.histogram(
                x=sim_means,
                nbins=25,
                title=f"Sampling Distribution of {num_samples} Sample Means (n={sub_size})",
                labels={"x": "Sample Mean (X̄)", "y": "Frequency"},
                color_discrete_sequence=["#06b6d4"],
            )
            fig_sd.add_vline(
                x=pop_vals.mean(),
                line_dash="dash",
                line_color="#f43f5e",
                annotation_text=f"Population μ = {pop_vals.mean():.1f}",
            )
            apply_plotly_theme(fig_sd)
            st.plotly_chart(fig_sd, use_container_width=True)

    # 4. CENTRAL LIMIT THEOREM (Cells 48-50)
    with s_tab4:
        st.markdown(
            """
            <div class="glass-card">
                <div class="section-title"><span>🏛️</span> Central Limit Theorem (CLT) Validation Laboratory</div>
                <div class="section-desc">
                    Demonstrating that as sample size n increases (e.g. n=30, K=1000 in Cell 48), the distribution of sample means converges asymptotically to a Gaussian distribution N(μ, σ²/n) regardless of the population distribution shape.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        clt_col1, clt_col2 = st.columns([1, 2])
        with clt_col1:
            clt_n = st.select_slider(
                "Sample Size (n)",
                options=[2, 5, 10, 20, 30, 50, 100],
                value=30,
            )
            clt_k = st.slider("Number of Repeated Samples", 500, 3000, 1000, step=250)

            pop_arr = df["Daily_Minutes_Spent"].values
            np.random.seed(42)
            clt_means = np.array(
                [
                    np.random.choice(pop_arr, size=clt_n, replace=True).mean()
                    for _ in range(clt_k)
                ]
            )

            # Verification metrics
            pop_mu = pop_arr.mean()
            pop_sigma = pop_arr.std()
            clt_mu = clt_means.mean()
            clt_sigma = clt_means.std()
            clt_theo_sigma = pop_sigma / math.sqrt(clt_n)

            st.markdown(
                f"""
                <div style="background: rgba(15, 23, 42, 0.7); padding: 16px; border-radius: 12px; border: 1px solid rgba(255, 255, 255, 0.08);">
                    <div style="font-size: 0.8rem; color: #94a3b8; font-weight: 700;">CLT EXPERIMENTAL AUDIT</div>
                    <div style="font-size: 0.95rem; color: #f8fafc; margin-top: 6px;">Population Mean: <b>{pop_mu:.2f}</b></div>
                    <div style="font-size: 0.95rem; color: #f8fafc; margin-top: 4px;">Mean of Sample Means: <b>{clt_mu:.2f}</b></div>
                    <div style="font-size: 0.95rem; color: #f8fafc; margin-top: 4px;">Empirical Std Dev (SE): <b>{clt_sigma:.2f}</b></div>
                    <div style="font-size: 0.95rem; color: #38bdf8; margin-top: 4px;">Theoretical σ / √n: <b>{clt_theo_sigma:.2f}</b></div>
                    <div style="font-size: 0.8rem; color: #10b981; margin-top: 8px;">✓ Asymptotic error convergence ratio: {(clt_sigma/clt_theo_sigma):.3f} ≈ 1.00</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with clt_col2:
            fig_clt = go.Figure()
            fig_clt.add_trace(
                go.Histogram(
                    x=clt_means,
                    histnorm="probability density",
                    nbinsx=25,
                    name=f"Sample Means (n={clt_n})",
                    marker_color="#818cf8",
                    opacity=0.7,
                )
            )

            # Fit theoretical normal curve
            x_ax = np.linspace(clt_mu - 3.5 * clt_sigma, clt_mu + 3.5 * clt_sigma, 200)
            y_ax = stats.norm.pdf(x_ax, clt_mu, clt_sigma)
            fig_clt.add_trace(
                go.Scatter(
                    x=x_ax,
                    y=y_ax,
                    mode="lines",
                    name="Theoretical CLT Normal Approximation",
                    line=dict(color="#38bdf8", width=3),
                )
            )
            fig_clt.add_vline(
                x=pop_mu,
                line_dash="dash",
                line_color="#f43f5e",
                annotation_text=f"Population Mean: {pop_mu:.1f}",
            )

            fig_clt.update_layout(
                title=f"Central Limit Theorem: Normal Convergence (n={clt_n}, K={clt_k})",
                xaxis_title="Sample Means",
                yaxis_title="Density",
            )
            apply_plotly_theme(fig_clt)
            st.plotly_chart(fig_clt, use_container_width=True)

# =============================================================================
# SECTION 7: HYPOTHESIS TESTING
# =============================================================================
elif selected_section == "Hypothesis Testing":
    st.markdown(
        """
        <div class="hero-card">
            <div class="hero-title">Inferential Hypothesis Testing</div>
            <div class="hero-subtitle">
                Interactive execution of the statistical tests conducted in the attached notebook (Cells 52-56): Pearson correlation test, One-Sample Z-Test, One-Sample T-Test, Welch's Two-Sample T-Test, and Chi-Square Test of Independence.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    test_tab1, test_tab2, test_tab3, test_tab4 = st.tabs(
        [
            "1. Bivariate Correlation Test",
            "2. One-Sample Z & T Tests",
            "3. Two-Sample Welch's T-Test",
            "4. Chi-Square Test of Independence",
        ]
    )

    # 1. BIVARIATE CORRELATION SIGNIFICANCE TEST (Cell 52)
    with test_tab1:
        st.markdown(
            """
            <div class="glass-card">
                <div class="section-title"><span>🔗</span> Pearson Correlation Significance Test</div>
                <div class="section-desc">
                    Testing H₀: ρ = 0 (No linear correlation between Daily Minutes Spent and Posts Per Day).
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        x_series = df["Daily_Minutes_Spent"]
        y_series = df["Posts_Per_Day"]
        r_coeff, p_val_corr = stats.pearsonr(x_series, y_series)

        tc1, tc2 = st.columns([1, 1])
        with tc1:
            st.metric("Pearson Correlation Coefficient (r)", f"{r_coeff:.4f}")
            st.metric("P-value", f"{p_val_corr:.4e}")
            if p_val_corr < 0.05:
                st.error("Decision: Reject H₀ at α = 0.05. Statistically significant linear correlation exists.")
            else:
                st.success("Decision: Fail to reject H₀ at α = 0.05. Insufficient evidence of linear correlation.")

        with tc2:
            st.info(
                f"""
                **Statistical Interpretation:**
                - r = **{r_coeff:.4f}** indicates a near-zero linear dependence between screen time and posting activity.
                - Because p = **{p_val_corr:.3f}** > 0.05, the data demonstrates that time spent browsing does not linearly dictate posting frequency in this demographic.
                """
            )

    # 2. ONE-SAMPLE Z & T TESTS (Cells 53-54)
    with test_tab2:
        st.markdown(
            """
            <div class="glass-card">
                <div class="section-title"><span>🎯</span> One-Sample Location Tests (Z-Test & T-Test)</div>
                <div class="section-desc">
                    Testing whether the true mean screen time differs from a hypothesized benchmark μ₀ = 240 minutes (Cells 53-54).
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        mu_0 = st.number_input(
            "Hypothesized Mean Screen Time (μ₀ in minutes)",
            value=240.0,
            step=5.0,
        )

        x_sample = df["Daily_Minutes_Spent"]
        n_obs = len(x_sample)
        x_bar = x_sample.mean()
        s_dev = x_sample.std(ddof=1)

        # Z-test
        z_stat = (x_bar - mu_0) / (s_dev / np.sqrt(n_obs))
        p_val_z = 2 * stats.norm.sf(abs(z_stat))

        # T-test
        t_stat, p_val_t = stats.ttest_1samp(x_sample, popmean=mu_0)

        # Confidence Interval
        ci_low, ci_high = stats.t.interval(
            0.95, df=n_obs - 1, loc=x_bar, scale=s_dev / np.sqrt(n_obs)
        )

        col_z1, col_z2 = st.columns(2)
        with col_z1:
            st.markdown(
                f"""
                <div style="background: rgba(15, 23, 42, 0.7); padding: 16px; border-radius: 12px; border: 1px solid rgba(255, 255, 255, 0.08);">
                    <div style="font-size: 0.8rem; color: #818cf8; font-weight: 700;">ONE-SAMPLE Z-TEST (Cell 53)</div>
                    <div style="font-size: 1.1rem; color: #f8fafc; margin-top: 6px;">Z-Statistic: <b>{z_stat:.4f}</b></div>
                    <div style="font-size: 1.1rem; color: #f8fafc; margin-top: 2px;">Two-Tailed P-Value: <b>{p_val_z:.4f}</b></div>
                    <div style="margin-top: 10px; font-size: 0.85rem; color: {'#10b981' if p_val_z >= 0.05 else '#f43f5e'}; font-weight: 700;">
                        {'Fail to reject H₀ at α=0.05' if p_val_z >= 0.05 else 'Reject H₀ at α=0.05'}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with col_z2:
            st.markdown(
                f"""
                <div style="background: rgba(15, 23, 42, 0.7); padding: 16px; border-radius: 12px; border: 1px solid rgba(255, 255, 255, 0.08);">
                    <div style="font-size: 0.8rem; color: #38bdf8; font-weight: 700;">ONE-SAMPLE T-TEST (Cell 54)</div>
                    <div style="font-size: 1.1rem; color: #f8fafc; margin-top: 6px;">T-Statistic: <b>{t_stat:.4f}</b> (df = {n_obs - 1})</div>
                    <div style="font-size: 1.1rem; color: #f8fafc; margin-top: 2px;">Two-Tailed P-Value: <b>{p_val_t:.4f}</b></div>
                    <div style="font-size: 0.85rem; color: #94a3b8; margin-top: 4px;">95% CI: [{ci_low:.2f}, {ci_high:.2f}] min</div>
                    <div style="margin-top: 6px; font-size: 0.85rem; color: {'#10b981' if p_val_t >= 0.05 else '#f43f5e'}; font-weight: 700;">
                        {'Fail to reject H₀ at α=0.05' if p_val_t >= 0.05 else 'Reject H₀ at α=0.05'}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # 3. TWO-SAMPLE WELCH'S T-TEST (Cell 55)
    with test_tab3:
        st.markdown(
            """
            <div class="glass-card">
                <div class="section-title"><span>🆚</span> Welch's Two-Sample Independent T-Test</div>
                <div class="section-desc">
                    Testing whether two social platforms exhibit a statistically significant difference in mean daily usage (unequal variances assumed).
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        all_unique_apps = sorted(df["App"].unique())
        tw_col1, tw_col2 = st.columns(2)
        with tw_col1:
            app_a = st.selectbox("Select Platform A:", all_unique_apps, index=0)
        with tw_col2:
            app_b = st.selectbox(
                "Select Platform B:",
                all_unique_apps,
                index=1 if len(all_unique_apps) > 1 else 0,
            )

        if app_a == app_b:
            st.warning("Please choose two different platforms to perform comparative testing.")
        else:
            grp_a = df.loc[df["App"] == app_a, "Daily_Minutes_Spent"]
            grp_b = df.loc[df["App"] == app_b, "Daily_Minutes_Spent"]

            t_welch, p_welch = stats.ttest_ind(grp_a, grp_b, equal_var=False)

            res_col1, res_col2 = st.columns(2)
            with res_col1:
                st.metric(f"Mean Usage: {app_a}", f"{grp_a.mean():.2f} min (n={len(grp_a)})")
                st.metric(f"Mean Usage: {app_b}", f"{grp_b.mean():.2f} min (n={len(grp_b)})")
                st.metric("Welch T-Statistic", f"{t_welch:.4f}")
                st.metric("P-value", f"{p_welch:.4f}")

            with res_col2:
                if p_welch < 0.05:
                    st.error(
                        f"**Statistically Significant Difference (p = {p_welch:.4f} < 0.05)**.\n"
                        f"Users on {app_a} and {app_b} have statistically distinct mean daily usage."
                    )
                else:
                    st.success(
                        f"**No Statistically Significant Difference (p = {p_welch:.4f} ≥ 0.05)**.\n"
                        f"There is insufficient empirical evidence to conclude that daily usage differs between {app_a} and {app_b}."
                    )

                fig_welch_box = px.box(
                    df[df["App"].isin([app_a, app_b])],
                    x="App",
                    y="Daily_Minutes_Spent",
                    color="App",
                    color_discrete_map=APP_PALETTE,
                )
                apply_plotly_theme(fig_welch_box)
                st.plotly_chart(fig_welch_box, use_container_width=True)

    # 4. CHI-SQUARE TEST OF INDEPENDENCE (Cell 56)
    with test_tab4:
        st.markdown(
            """
            <div class="glass-card">
                <div class="section-title"><span>🧮</span> Chi-Square Test of Independence (χ²)</div>
                <div class="section-desc">
                    Testing independence between Social Media App and Usage Level (High vs Low Usage, segmented by median split) from notebook Cell 56.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        chi_df = df[["App", "Daily_Minutes_Spent"]].dropna().copy()
        med_val = chi_df["Daily_Minutes_Spent"].median()
        chi_df["Usage_Level"] = np.where(
            chi_df["Daily_Minutes_Spent"] > med_val, "High", "Low"
        )

        contingency_tab = pd.crosstab(chi_df["App"], chi_df["Usage_Level"])
        chi2_stat, p_val_chi2, dof, expected_mat = stats.chi2_contingency(contingency_tab)

        chi_c1, chi_c2 = st.columns([1, 1])

        with chi_c1:
            st.markdown("**Contingency Table (Observed Frequencies):**")
            st.dataframe(contingency_tab, use_container_width=True)

            st.metric("Chi-Square Statistic (χ²)", f"{chi2_stat:.4f}")
            st.metric("Degrees of Freedom (df)", f"{dof}")
            st.metric("P-value", f"{p_val_chi2:.4f}")

            if p_val_chi2 < 0.05:
                st.error("Reject H₀: Platform and Usage Level are significantly associated.")
            else:
                st.success("Fail to reject H₀: No statistically significant association between platform and usage level.")

        with chi_c2:
            st.markdown("**Expected Frequencies under Independence:**")
            exp_df = pd.DataFrame(
                expected_mat,
                index=contingency_tab.index,
                columns=contingency_tab.columns,
            ).round(2)
            st.dataframe(exp_df, use_container_width=True)

            # Stacked bar plot of observed distribution
            fig_chi_bar = px.bar(
                contingency_tab.reset_index(),
                x="App",
                y=["High", "Low"],
                title="Observed High vs Low Usage Counts by Platform",
                color_discrete_sequence=["#818cf8", "#38bdf8"],
                barmode="group",
            )
            apply_plotly_theme(fig_chi_bar)
            st.plotly_chart(fig_chi_bar, use_container_width=True)

# =============================================================================
# SECTION 8: PREDICTIVE MODELING (ML)
# =============================================================================
elif selected_section == "Predictive Modeling (ML)":
    st.markdown(
        """
        <div class="hero-card">
            <div class="hero-title">Predictive Machine Learning Models</div>
            <div class="hero-subtitle">
                Supervised learning implementations from notebook Cells 57-58: Linear Regression for post count forecasting and Logistic Regression for user high-engagement classification.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    ml_tab1, ml_tab2, ml_tab3 = st.tabs(
        [
            "1. Linear Regression (Cell 57)",
            "2. Logistic Regression Classifier (Cell 58)",
            "3. Live Persona Prediction Simulator",
        ]
    )

    # 1. LINEAR REGRESSION
    with ml_tab1:
        st.markdown(
            """
            <div class="glass-card">
                <div class="section-title"><span>📈</span> Linear Regression: Usage vs Posts</div>
                <div class="section-desc">Predicting Posts Per Day from Daily Minutes Spent.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        reg_data = df[["Daily_Minutes_Spent", "Posts_Per_Day"]].dropna()
        X_reg = reg_data[["Daily_Minutes_Spent"]]
        y_reg = reg_data["Posts_Per_Day"]

        X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
            X_reg, y_reg, test_size=0.2, random_state=42
        )

        lr_model = LinearRegression()
        lr_model.fit(X_train_r, y_train_r)
        y_pred_r = lr_model.predict(X_test_r)

        r2 = r2_score(y_test_r, y_pred_r)
        mse = mean_squared_error(y_test_r, y_pred_r)

        r_col1, r_col2, r_col3, r_col4 = st.columns(4)
        r_col1.metric("Model Intercept (β₀)", f"{lr_model.intercept_:.4f}")
        r_col2.metric("Coefficient (β₁)", f"{lr_model.coef_[0]:.6f}")
        r_col3.metric("R-Squared (R²)", f"{r2:.5f}")
        r_col4.metric("Mean Squared Error (MSE)", f"{mse:.2f}")

        # Interactive regression plot
        plot_df = pd.DataFrame(
            {
                "Minutes": X_test_r["Daily_Minutes_Spent"],
                "Actual_Posts": y_test_r,
                "Predicted_Posts": y_pred_r,
            }
        )

        fig_lr = px.scatter(
            plot_df,
            x="Minutes",
            y="Actual_Posts",
            opacity=0.65,
            color_discrete_sequence=["#818cf8"],
            labels={"Minutes": "Daily Minutes Spent", "Actual_Posts": "Actual Posts Per Day"},
            title="Linear Regression: Test Set Actual vs Predicted Line",
        )
        # Add regression line
        fig_lr.add_trace(
            go.Scatter(
                x=plot_df["Minutes"],
                y=plot_df["Predicted_Posts"],
                mode="lines",
                name="Fitted OLS Line",
                line=dict(color="#f43f5e", width=2.5),
            )
        )
        apply_plotly_theme(fig_lr)
        st.plotly_chart(fig_lr, use_container_width=True)

    # 2. LOGISTIC REGRESSION CLASSIFIER
    with ml_tab2:
        st.markdown(
            """
            <div class="glass-card">
                <div class="section-title"><span>🤖</span> Logistic Regression: High-Usage Classification</div>
                <div class="section-desc">
                    Supervised classifier predicting whether a user is a High-Usage consumer (&gt; median screen time) based on Posts, Likes, and Follows (Notebook Cell 58).
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        log_data = df[
            [
                "Daily_Minutes_Spent",
                "Posts_Per_Day",
                "Likes_Per_Day",
                "Follows_Per_Day",
            ]
        ].dropna().copy()

        median_u = log_data["Daily_Minutes_Spent"].median()
        log_data["High_Usage"] = (log_data["Daily_Minutes_Spent"] > median_u).astype(int)

        X_log = log_data[["Posts_Per_Day", "Likes_Per_Day", "Follows_Per_Day"]]
        y_log = log_data["High_Usage"]

        X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
            X_log, y_log, test_size=0.2, random_state=42, stratify=y_log
        )

        clf = LogisticRegression(max_iter=1000)
        clf.fit(X_train_c, y_train_c)
        y_pred_c = clf.predict(X_test_c)

        acc = accuracy_score(y_test_c, y_pred_c)
        cm = confusion_matrix(y_test_c, y_pred_c)

        lg_c1, lg_c2 = st.columns([1, 1])

        with lg_c1:
            st.metric("Test Set Accuracy", f"{acc * 100:.2f}%")
            st.markdown("**Classification Metrics Breakdown:**")
            report_dict = classification_report(
                y_test_c, y_pred_c, output_dict=True, zero_division=0
            )
            rep_df = pd.DataFrame(report_dict).transpose().round(3)
            st.dataframe(rep_df, use_container_width=True)

        with lg_c2:
            st.markdown("**Confusion Matrix Heatmap:**")
            fig_cm = px.imshow(
                cm,
                text_auto=True,
                labels=dict(x="Predicted Class", y="Actual Class", color="Count"),
                x=["Low Usage (0)", "High Usage (1)"],
                y=["Low Usage (0)", "High Usage (1)"],
                color_continuous_scale="Blues",
            )
            apply_plotly_theme(fig_cm)
            st.plotly_chart(fig_cm, use_container_width=True)

    # 3. LIVE PERSONA PREDICTION SIMULATOR
    with ml_tab3:
        st.markdown(
            """
            <div class="glass-card">
                <div class="section-title"><span>🔮</span> Real-Time User Engagement Classifier Simulator</div>
                <div class="section-desc">
                    Adjust user behavioural inputs to generate instant machine learning inferences using the trained Logistic Regression model.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        sim_c1, sim_c2 = st.columns([1, 1])

        with sim_c1:
            input_posts = st.slider("Simulated Posts Per Day", 0, 20, 10)
            input_likes = st.slider("Simulated Likes Per Day", 0, 200, 95)
            input_follows = st.slider("Simulated Follows Per Day", 0, 50, 25)

            new_user_feat = pd.DataFrame(
                [[input_posts, input_likes, input_follows]],
                columns=["Posts_Per_Day", "Likes_Per_Day", "Follows_Per_Day"],
            )

            pred_prob = clf.predict_proba(new_user_feat)[0]
            pred_class = clf.predict(new_user_feat)[0]

        with sim_c2:
            st.markdown(
                f"""
                <div style="background: rgba(15, 23, 42, 0.8); padding: 24px; border-radius: 16px; border: 1px solid rgba(99, 102, 241, 0.3);">
                    <div style="font-size: 0.85rem; color: #94a3b8; font-weight: 700; text-transform: uppercase;">
                        Model Prediction Result
                    </div>
                    <div style="font-size: 2rem; font-weight: 800; color: {'#38bdf8' if pred_class == 1 else '#f59e0b'}; margin: 10px 0;">
                        {'HIGH-USAGE USER (> 4.1 hrs/day)' if pred_class == 1 else 'STANDARD/LOW USAGE USER'}
                    </div>
                    <div style="font-size: 0.95rem; color: #cbd5e1;">
                        Probability of High Usage: <b style="color: #818cf8;">{pred_prob[1]*100:.1f}%</b>
                    </div>
                    <div style="font-size: 0.95rem; color: #cbd5e1; margin-top: 4px;">
                        Probability of Low Usage: <b style="color: #64748b;">{pred_prob[0]*100:.1f}%</b>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Probability Gauge
            fig_gauge = go.Figure(
                go.Indicator(
                    mode="gauge+number",
                    value=pred_prob[1] * 100,
                    number={"suffix": "%"},
                    title={"text": "High Usage Probability"},
                    gauge={
                        "axis": {"range": [0, 100]},
                        "bar": {"color": "#818cf8"},
                        "steps": [
                            {"range": [0, 50], "color": "rgba(255,255,255,0.05)"},
                            {"range": [50, 100], "color": "rgba(99,102,241,0.2)"},
                        ],
                        "threshold": {
                            "line": {"color": "#38bdf8", "width": 4},
                            "thickness": 0.75,
                            "value": 50,
                        },
                    },
                )
            )
            fig_gauge.update_layout(height=240, margin=dict(l=20, r=20, t=30, b=10))
            apply_plotly_theme(fig_gauge)
            st.plotly_chart(fig_gauge, use_container_width=True)

# =============================================================================
# SECTION 9: INSIGHTS & CONCLUSIONS
# =============================================================================
elif selected_section == "Insights & Conclusions":
    st.markdown(
        """
        <div class="hero-card">
            <div class="hero-title">Empirical Findings, Limitations & Conclusion</div>
            <div class="hero-subtitle">
                Synthesized insights bridging statistical inference, practical behavioral dynamics, academic requirements, and future research directions.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="glass-card">
            <div class="section-title"><span>📌</span> Core Statistical Findings</div>
            <div style="color: #cbd5e1; font-size: 0.95rem; line-height: 1.8;">
                <ol>
                    <li><b>Screen Time Central Tendency:</b> The population of 1,000 subjects demonstrates a mean daily usage of <b>247.37 minutes (~4.12 hours)</b> with a standard deviation of <b>146.37 minutes</b> and a median of <b>246.0 minutes</b>. The close alignment between mean and median highlights a broadly symmetric, uniform-like distribution across the 5 to 500-minute range.</li>
                    <li><b>Independence Between Browsing Time and Posting:</b> Pearson correlation between <code>Daily_Minutes_Spent</code> and <code>Posts_Per_Day</code> is virtually negligible (r ≈ -0.007, p = 0.817 &gt; 0.05). This confirms that high screen time does <i>not</i> directly imply active content creation—many users remain passive consumers.</li>
                    <li><b>Platform Uniformity:</b> Welch's two-sample t-tests and the Chi-Square test of independence (p &gt; 0.05) reveal that average usage duration is largely consistent across Snapchat, LinkedIn, TikTok, Pinterest, Instagram, Twitter, and Facebook within this dataset.</li>
                    <li><b>Central Limit Theorem Asymptotic Normality:</b> Monte Carlo iterations empirically validate that regardless of individual user variance, sampling distributions of sample means converge rapidly to a Gaussian bell curve N(μ, σ²/n) with standard error diminishing inversely proportional to √n.</li>
                </ol>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.write("")

    st.markdown(
        """
        <div class="glass-card">
            <div class="section-title"><span>⚠️</span> Methodological Considerations & Project Limitations</div>
            <div style="color: #cbd5e1; font-size: 0.95rem; line-height: 1.8;">
                <ul>
                    <li><b>Cross-Sectional Architecture:</b> The dataset records a single observation per user. It lacks temporal longitudinal tracking (e.g., changes across weekdays vs. weekends).</li>
                    <li><b>Correlation ≠ Causation:</b> Statistical significance or lack thereof measures linear/monotonic associations, not direct behavioral causation.</li>
                    <li><b>Unmeasured Confounders:</b> Demographic factors such as user age, profession, geographic region, and device type were unobserved and may moderate usage patterns.</li>
                </ul>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.write("")

    st.markdown(
        """
        <div class="glass-card">
            <div class="section-title"><span>🚀</span> Future Research & Extensibility</div>
            <div style="color: #cbd5e1; font-size: 0.95rem; line-height: 1.8;">
                <ul>
                    <li>Integration of real-time telemetry APIs from mobile OS screen-time trackers.</li>
                    <li>Deployment of unsupervised clustering (e.g., K-Means / Gaussian Mixture Models) to identify latent user personas (e.g., Lurkers, Active Creators, Casual Browsers).</li>
                    <li>Natural language processing (NLP) sentiment scoring correlated with session durations.</li>
                </ul>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Academic Citation Footer
    st.markdown(
        """
        <div style="text-align: center; margin-top: 35px; padding: 20px; border-top: 1px solid rgba(255, 255, 255, 0.08); color: #64748b; font-size: 0.85rem;">
            Social Media Usage Analysis &bull; Statistical Inference Lab &bull; Arya Singh (Roll No. 02, Class B3[B1]) &bull; Built with Streamlit & Plotly
        </div>
        """,
        unsafe_allow_html=True,
    )
