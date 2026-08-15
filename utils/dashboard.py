"""
Main Dashboard UI Component for AI-Based Project Risk Forecasting System
Fetches real-time data from database and renders the enterprise dashboard with Dark Magenta & Neon Rose aesthetic.
"""

import os
import streamlit as st
import pandas as pd
from utils.database_client import get_user_dashboard_metrics


def render_top_navigation():
    """Renders single-line Top Navigation for authenticated dashboard workspace supporting Dark Magenta & Neon Rose aesthetic."""
    user = st.session_state.get("user_doc", {})

    st.markdown("""
        <style>
            /* -----------------------------------------------------------------
               DARK MAGENTA & NEON ROSE DESIGN TOKENS FOR DASHBOARD WORKSPACE
            ----------------------------------------------------------------- */
            :root {
                --bg-canvas: #140713;
                --bg-card: #1d0c1b;
                --border-color: rgba(225, 29, 126, 0.3);
                --text-primary: #ffffff;
                --text-secondary: #f472b6;
                --card-shadow: 0 10px 30px rgba(0, 0, 0, 0.5);
                --nav-btn-bg: linear-gradient(135deg, #2a0e26, #180616);
                --nav-btn-hover: linear-gradient(135deg, #e11d74, #9f1239);
                --active-tab-bg: linear-gradient(135deg, #e11d74, #9f1239);
            }

            @media (prefers-color-scheme: dark) {
                :root {
                    --bg-canvas: #140713;
                    --bg-card: #1d0c1b;
                    --border-color: rgba(225, 29, 126, 0.35);
                    --text-primary: #ffffff;
                    --text-secondary: #f472b6;
                    --card-shadow: 0 12px 35px rgba(0, 0, 0, 0.6);
                    --nav-btn-bg: linear-gradient(135deg, #2a0e26, #180616);
                    --nav-btn-hover: linear-gradient(135deg, #e11d74, #9f1239);
                    --active-tab-bg: linear-gradient(135deg, #e11d74, #9f1239);
                }
            }

            /* Responsive Canvas and High-Contrast Typography */
            .stApp, [data-testid="stAppViewContainer"] {
                background-color: var(--bg-canvas) !important;
                background: radial-gradient(circle at 50% 12%, rgba(225, 29, 126, 0.22) 0%, rgba(20, 7, 19, 0.98) 65%, #140713 100%) !important;
                color: var(--text-primary) !important;
            }

            h1, h2, h3, h4, h5, h6 {
                color: var(--text-primary) !important;
            }

            /* Dashboard Top Navigation Container */
            .dashboard-navbar-container {
                display: flex;
                justify-content: space-between;
                align-items: center;
                background: rgba(29, 12, 27, 0.85) !important;
                border: 1px solid var(--border-color) !important;
                border-radius: 16px;
                padding: 14px 28px;
                margin-bottom: 32px;
                box-shadow: var(--card-shadow);
                backdrop-filter: blur(14px);
            }

            .dashboard-brand {
                font-family: 'Outfit', sans-serif;
                font-size: 1.25rem;
                font-weight: 800;
                color: #ffffff !important;
                display: flex;
                align-items: center;
                gap: 8px;
                white-space: nowrap;
            }

            /* Streamlit Button Overrides with High Contrast White Text */
            .stButton > button,
            .stButton > button * {
                background: var(--nav-btn-bg) !important;
                color: #ffffff !important;
                -webkit-text-fill-color: #ffffff !important;
                opacity: 1 !important;
                font-weight: 600 !important;
                font-size: 0.88rem !important;
                letter-spacing: 0.2px !important;
            }

            .stButton > button {
                border-radius: 8px !important;
                height: 40px !important;
                min-height: 40px !important;
                max-height: 40px !important;
                padding: 0 16px !important;
                border: 1px solid rgba(225, 29, 126, 0.25) !important;
                white-space: nowrap !important;
                display: inline-flex !important;
                align-items: center !important;
                justify-content: center !important;
                width: 100% !important;
                box-shadow: 0 3px 10px rgba(0, 0, 0, 0.4) !important;
                transition: all 0.2s ease-in-out !important;
            }

            .stButton > button:hover,
            .stButton > button:hover * {
                background: var(--nav-btn-hover) !important;
                border-color: #e11d74 !important;
                color: #ffffff !important;
                -webkit-text-fill-color: #ffffff !important;
                box-shadow: 0 5px 15px rgba(225, 29, 126, 0.4) !important;
            }

            /* Active Tab Highlight Style */
            .active-nav-btn button,
            .active-nav-btn button * {
                background: var(--active-tab-bg) !important;
                border-color: #fb7185 !important;
                color: #ffffff !important;
                -webkit-text-fill-color: #ffffff !important;
            }

            .active-nav-btn button {
                box-shadow: 0 4px 18px rgba(225, 29, 126, 0.45) !important;
            }

            /* Metric Card Container Overrides */
            [data-testid="stMetric"] {
                background: rgba(29, 12, 27, 0.85) !important;
                border: 1px solid var(--border-color) !important;
                border-radius: 14px !important;
                padding: 16px 20px !important;
                box-shadow: var(--card-shadow) !important;
                backdrop-filter: blur(12px);
            }

            [data-testid="stMetricValue"], [data-testid="stMetricValue"] * {
                color: #ffffff !important;
                -webkit-text-fill-color: #ffffff !important;
                font-size: 2rem !important;
                font-weight: 800 !important;
            }

            [data-testid="stMetricLabel"], [data-testid="stMetricLabel"] * {
                color: #f472b6 !important;
                -webkit-text-fill-color: #f472b6 !important;
                font-size: 0.9rem !important;
                font-weight: 700 !important;
            }
        </style>
    """, unsafe_allow_html=True)

    nav_cols = st.columns([3, 1, 1.1, 1.1, 1, 1, 1], vertical_alignment="center")
    active_tab = st.session_state.get("active_tab", "dashboard")

    with nav_cols[0]:
        st.markdown('<div class="dashboard-brand">🛡️ AI Risk Forecasting</div>', unsafe_allow_html=True)

    with nav_cols[1]:
        if active_tab == "dashboard":
            st.markdown('<div class="active-nav-btn">', unsafe_allow_html=True)
        if st.button("Dashboard", key="nav_dash"):
            st.session_state.active_tab = "dashboard"
            st.rerun()
        if active_tab == "dashboard":
            st.markdown('</div>', unsafe_allow_html=True)

    with nav_cols[2]:
        if active_tab == "risk_analysis":
            st.markdown('<div class="active-nav-btn">', unsafe_allow_html=True)
        if st.button("Risk Analysis", key="nav_analysis"):
            st.session_state.active_tab = "risk_analysis"
            st.rerun()
        if active_tab == "risk_analysis":
            st.markdown('</div>', unsafe_allow_html=True)

    with nav_cols[3]:
        if active_tab == "visualization":
            st.markdown('<div class="active-nav-btn">', unsafe_allow_html=True)
        if st.button("Visualization", key="nav_viz"):
            st.session_state.active_tab = "visualization"
            st.rerun()
        if active_tab == "visualization":
            st.markdown('</div>', unsafe_allow_html=True)

    with nav_cols[4]:
        if active_tab == "history":
            st.markdown('<div class="active-nav-btn">', unsafe_allow_html=True)
        if st.button("History", key="nav_hist"):
            st.session_state.active_tab = "history"
            st.rerun()
        if active_tab == "history":
            st.markdown('</div>', unsafe_allow_html=True)

    with nav_cols[5]:
        if active_tab == "profile":
            st.markdown('<div class="active-nav-btn">', unsafe_allow_html=True)
        if st.button("Profile", key="nav_prof"):
            st.session_state.active_tab = "profile"
            st.rerun()
        if active_tab == "profile":
            st.markdown('</div>', unsafe_allow_html=True)

    with nav_cols[6]:
        if st.button("Logout", key="nav_logout"):
            st.session_state.authenticated = False
            st.session_state.user_doc = None
            st.session_state.current_page = "landing"
            st.rerun()

    st.markdown("<div style='margin-bottom: 24px;'></div>", unsafe_allow_html=True)


def render_main_dashboard():
    """Renders the Main Dashboard overview page."""

    # Ensure top navigation is displayed
    render_top_navigation()

    user = st.session_state.get("user_doc", {})
    user_id = user.get("_id", "")
    user_name = f"{user.get('first_name', 'User')} {user.get('last_name', '')}".strip()

    # Load real-time metrics
    metrics = get_user_dashboard_metrics(user_id)

    total_projects = metrics["total_projects"]
    high_risk = metrics["high_risk_count"]
    medium_risk = metrics["medium_risk_count"]
    low_risk = metrics["low_risk_count"]
    avg_risk_pct = metrics["avg_risk_score_pct"]
    predictions = metrics["predictions"]

    # -------------------------------------------------------------------------
    # WELCOME SECTION (INTEGRATED ACTION CARD)
    # -------------------------------------------------------------------------
    st.markdown(f"""
        <div style="background: rgba(29, 12, 27, 0.85); border: 1.5px solid var(--border-color); border-radius: 18px; padding: 32px 32px 24px 32px; margin-bottom: 32px; box-shadow: var(--card-shadow); backdrop-filter: blur(14px);">
            <h1 style="font-size: 2rem; font-weight: 800; color: #ffffff !important; margin-bottom: 8px; letter-spacing: -0.5px;">
                Welcome back, {user_name}!
            </h1>
            <p style="font-size: 1rem; color: #f472b6 !important; margin-bottom: 20px; max-width: 780px; line-height: 1.5;">
                Your project risk overview at a glance. Analyze your projects, understand potential risks, and make informed decisions.
            </p>
        </div>
    """, unsafe_allow_html=True)

    w_col1, w_col2, w_col3 = st.columns([1.5, 2, 1.5])
    with w_col2:
        if st.button("🚀 Start Risk Analysis", key="btn_start_analysis", use_container_width=True):
            st.session_state.active_tab = "risk_analysis"
            st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # PROJECT OVERVIEW (5 KPI METRICS CARDS)
    # -------------------------------------------------------------------------
    st.markdown("""
        <div style="margin-bottom: 16px;">
            <h3 style="font-size: 1.3rem; font-weight: 800; color: #ffffff !important;">PROJECT OVERVIEW</h3>
        </div>
    """, unsafe_allow_html=True)

    kpi_col1, kpi_col2, kpi_col3, kpi_col4, kpi_col5 = st.columns(5)

    with kpi_col1:
        st.metric("Total Projects", total_projects, help="Projects analyzed through the system.")
    with kpi_col2:
        st.metric("High Risk", high_risk, help="Projects requiring immediate attention.")
    with kpi_col3:
        st.metric("Medium Risk", medium_risk, help="Projects that should be monitored closely.")
    with kpi_col4:
        st.metric("Low Risk", low_risk, help="Projects with relatively lower risk.")
    with kpi_col5:
        st.metric("Average Risk", avg_risk_pct, help="Average predicted risk score across analyzed projects.")

    st.markdown("<br><br>", unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # PROJECT RISK DISTRIBUTION
    # -------------------------------------------------------------------------
    st.markdown("""
        <div style="margin-bottom: 16px;">
            <h3 style="font-size: 1.3rem; font-weight: 800; color: #ffffff !important;">PROJECT RISK DISTRIBUTION</h3>
        </div>
    """, unsafe_allow_html=True)

    if total_projects == 0:
        st.markdown("""
            <div style="background: rgba(29, 12, 27, 0.85); border: 1.5px solid var(--border-color); border-radius: 16px; padding: 36px; text-align: center; box-shadow: var(--card-shadow); backdrop-filter: blur(14px);">
                <div style="font-size: 2.2rem; margin-bottom: 8px;">📊</div>
                <h4 style="font-size: 1.15rem; font-weight: 800; color: #ffffff !important; margin-bottom: 6px;">No project analysis available yet.</h4>
                <p style="font-size: 0.95rem; color: #f472b6 !important;">Start your first risk analysis to see your project risk distribution.</p>
            </div>
        """, unsafe_allow_html=True)
    else:
        dist_df = pd.DataFrame({
            "Risk Level": ["High Risk", "Medium Risk", "Low Risk"],
            "Count": [high_risk, medium_risk, low_risk]
        })
        st.bar_chart(dist_df.set_index("Risk Level"))

    st.markdown("<br><br>", unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # RECENT PROJECT ANALYSIS TABLE
    # -------------------------------------------------------------------------
    st.markdown("""
        <div style="margin-bottom: 16px;">
            <h3 style="font-size: 1.3rem; font-weight: 800; color: #ffffff !important;">RECENT PROJECT ANALYSIS</h3>
        </div>
    """, unsafe_allow_html=True)

    if total_projects == 0:
        st.markdown("""
            <div style="background: rgba(29, 12, 27, 0.85); border: 1.5px solid var(--border-color); border-radius: 16px; padding: 36px; text-align: center; box-shadow: var(--card-shadow); backdrop-filter: blur(14px);">
                <div style="font-size: 2.2rem; margin-bottom: 8px;">📋</div>
                <h4 style="font-size: 1.15rem; font-weight: 800; color: #ffffff !important; margin-bottom: 6px;">No projects analyzed yet.</h4>
            </div>
        """, unsafe_allow_html=True)
    else:
        table_data = []
        for p in predictions[:5]:
            table_data.append({
                "Project Name": p.get("project_name", "N/A"),
                "Project Type": p.get("input_features", {}).get("project_type", "Software"),
                "Risk Level": p.get("risk_level", "Medium"),
                "Risk Score": f"{p.get('risk_score', 0.0)}%",
                "Analyzed On": p.get("analyzed_at", "N/A")
            })
        st.table(pd.DataFrame(table_data))
