"""
Prediction History Component for AI-Based Project Risk Forecasting System
Implements Search & Filter, History Table, Empty State, and Detailed Project Modal View.
"""

import streamlit as st
import pandas as pd
from utils.database_client import get_user_predictions


def render_history_page():
    """Renders Project History with search, filter, sorting, table view, and detailed inspection."""
    user = st.session_state.get("user_doc", {})
    user_id = user.get("_id", "")

    # Main Header
    st.markdown("""
        <div style="margin-bottom: 24px;">
            <h2 style="font-size: 1.8rem; font-weight: 800; color: var(--text-primary);">PROJECT HISTORY</h2>
            <p style="color: var(--text-secondary); font-size: 1rem;">
                View and manage your previous project risk analyses.
            </p>
        </div>
    """, unsafe_allow_html=True)

    # Fetch real-time user predictions
    predictions = get_user_predictions(user_id)

    # -------------------------------------------------------------------------
    # EMPTY STATE (WHEN USER HASN'T ANALYZED ANY PROJECTS YET)
    # -------------------------------------------------------------------------
    if not predictions:
        with st.container():
            st.markdown("""
                <div style="background: var(--bg-card); border: 1px solid var(--border-color); border-radius: 18px; padding: 40px; text-align: center; margin-top: 10px; margin-bottom: 24px; box-shadow: var(--card-shadow);">
                    <div style="font-size: 3rem; margin-bottom: 12px;">📂</div>
                    <h3 style="font-size: 1.4rem; font-weight: 800; color: var(--text-primary) !important; margin-bottom: 8px;">No Project Analyses Yet</h3>
                    <p style="color: var(--text-secondary) !important; font-size: 1rem; margin-bottom: 6px;">You haven't analyzed any projects yet.</p>
                    <p style="color: var(--text-secondary) !important; opacity: 0.8; font-size: 0.9rem; margin-bottom: 20px;">Your completed project risk analyses will appear here after you perform a risk analysis.</p>
                </div>
            """, unsafe_allow_html=True)

            h_col1, h_col2, h_col3 = st.columns([1.5, 2, 1.5])
            with h_col2:
                if st.button("🚀 Start Your First Analysis", key="btn_first_analysis", use_container_width=True):
                    st.session_state.active_tab = "risk_analysis"
                    st.rerun()
        return

    # -------------------------------------------------------------------------
    # SEARCH & FILTER CONTROLS
    # -------------------------------------------------------------------------
    st.markdown("### SEARCH & FILTER")

    f_col1, f_col2, f_col3 = st.columns([2, 1, 1])
    with f_col1:
        search_query = st.text_input("Search Project", placeholder="Search by project name...", label_visibility="visible")
    with f_col2:
        filter_risk = st.selectbox("Risk Level", ["All Risk Levels", "High Risk", "Medium Risk", "Low Risk", "Critical Risk"])
    with f_col3:
        sort_by = st.selectbox("Sort By", ["Most Recent", "Oldest First", "Highest Risk", "Lowest Risk"])

    # Apply Filtering
    filtered_preds = list(predictions)

    if search_query.strip():
        q = search_query.strip().lower()
        filtered_preds = [p for p in filtered_preds if q in p.get("project_name", "").lower()]

    if filter_risk != "All Risk Levels":
        target_lvl = filter_risk.replace(" Risk", "").lower()
        filtered_preds = [p for p in filtered_preds if target_lvl in str(p.get("risk_level", "")).lower()]

    # Apply Sorting
    if sort_by == "Most Recent":
        filtered_preds.sort(key=lambda x: x.get("analyzed_at", ""), reverse=True)
    elif sort_by == "Oldest First":
        filtered_preds.sort(key=lambda x: x.get("analyzed_at", ""), reverse=False)
    elif sort_by == "Highest Risk":
        filtered_preds.sort(key=lambda x: float(x.get("risk_score", 0.0)), reverse=True)
    elif sort_by == "Lowest Risk":
        filtered_preds.sort(key=lambda x: float(x.get("risk_score", 0.0)), reverse=False)

    st.markdown("<br>", unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # ANALYSIS HISTORY TABLE
    # -------------------------------------------------------------------------
    st.markdown("### ANALYSIS HISTORY")

    if not filtered_preds:
        st.info("🔍 No matching project analyses found for the selected filters.")
        return

    color_map = {
        "Low": "#10b981",
        "Medium": "#f59e0b",
        "High": "#ef4444",
        "Critical": "#dc2626"
    }

    # Render History Table Cards
    for idx, item in enumerate(filtered_preds):
        p_name = item.get("project_name", "Untitled")
        p_type = item.get("input_features", {}).get("project_type", "Software Development")
        r_level = item.get("risk_level", "Medium")
        r_score = item.get("risk_score", 0.0)
        date_str = item.get("analyzed_at", "N/A")
        badge_c = color_map.get(r_level, "#ef4444")

        # Table Row Layout
        r_col1, r_col2, r_col3, r_col4, r_col5, r_col6 = st.columns([2.5, 2, 1.5, 1.2, 2, 1.5], vertical_alignment="center")

        with r_col1:
            st.markdown(f"**{p_name}**")
        with r_col2:
            st.markdown(f"<span style='color: var(--text-secondary);'>{p_type}</span>", unsafe_allow_html=True)
        with r_col3:
            st.markdown(f"<span style='background: {badge_c}; color: #ffffff; font-weight: 700; padding: 3px 12px; border-radius: 12px; font-size: 0.82rem;'>{r_level} Risk</span>", unsafe_allow_html=True)
        with r_col4:
            st.markdown(f"**{r_score}%**")
        with r_col5:
            st.markdown(f"<span style='color: var(--text-secondary); font-size: 0.85rem;'>{date_str}</span>", unsafe_allow_html=True)
        with r_col6:
            if st.button("View Details", key=f"btn_details_{idx}"):
                st.session_state.selected_history_project = item

        st.markdown("<hr style='border-color: var(--border-color); margin: 8px 0;'>", unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # PROJECT DETAILS MODAL CARD (WHEN USER SELECTS A PROJECT)
    # -------------------------------------------------------------------------
    if "selected_history_project" in st.session_state and st.session_state.selected_history_project:
        selected_p = st.session_state.selected_history_project
        s_name = selected_p.get("project_name", "Untitled")
        s_features = selected_p.get("input_features", {})
        s_type = s_features.get("project_type", "Software Development")
        s_sector = s_features.get("industry_sector", "IT")
        s_level = selected_p.get("risk_level", "Medium")
        s_score = selected_p.get("risk_score", 0.0)
        s_date = selected_p.get("analyzed_at", "N/A")
        badge_color = color_map.get(s_level, "#ef4444")

        st.markdown("<br><br>", unsafe_allow_html=True)
        st.markdown("### PROJECT DETAILS")

        st.markdown(f"""
            <div style="background: var(--bg-card); border: 2px solid {badge_color}; border-radius: 18px; padding: 32px; box-shadow: var(--card-shadow);">
                <div style="font-size: 1.1rem; color: var(--text-primary); margin-bottom: 10px;">
                    <strong>Project Name:</strong> {s_name}
                </div>
                <div style="font-size: 1.1rem; color: var(--text-primary); margin-bottom: 10px;">
                    <strong>Project Type:</strong> {s_type}
                </div>
                <div style="font-size: 1.1rem; color: var(--text-primary); margin-bottom: 16px;">
                    <strong>Industry Sector:</strong> {s_sector}
                </div>
                <hr style="border-color: var(--border-color); margin: 16px 0;">
                <div style="font-size: 1.1rem; color: var(--text-primary); margin-bottom: 10px; display: flex; align-items: center; gap: 10px;">
                    <strong>Predicted Risk Level:</strong>
                    <span style="background: {badge_color}; color: #ffffff; font-weight: 800; padding: 4px 16px; border-radius: 16px; font-size: 0.95rem;">
                        {s_level} Risk
                    </span>
                </div>
                <div style="font-size: 1.1rem; color: var(--text-primary); margin-bottom: 16px;">
                    <strong>Risk Score:</strong> <span style="color: {badge_color}; font-weight: 800; font-size: 1.4rem;">{s_score}%</span>
                </div>
                <div style="font-size: 0.92rem; color: var(--text-secondary); margin-bottom: 20px;">
                    <strong>Analyzed On:</strong> {s_date}
                </div>
            </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        d_col1, d_col2, d_col3 = st.columns([1.5, 2, 1.5])
        with d_col2:
            if st.button("📊 View Visualization", key="btn_hist_viz", use_container_width=True):
                st.session_state.active_tab = "visualization"
                st.rerun()
