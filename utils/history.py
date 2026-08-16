"""
Prediction History Component for AI-Based Project Risk Forecasting System
Implements Search & Filter, History Table, Individual & Batch Deletion, and Detailed Project Inspection.
"""

import streamlit as st
import pandas as pd
from utils.api_client import get_user_predictions, delete_prediction, delete_all_predictions


def render_history_page():
    """Renders Project History with search, filter, sorting, table view, deletion, and detailed inspection."""
    user = st.session_state.get("user_doc", {})
    user_id = user.get("_id", "")

    # Main Header with Clear All Action Button
    h_top1, h_top2 = st.columns([3, 1], vertical_alignment="center")
    with h_top1:
        st.markdown("""
            <div style="margin-bottom: 16px;">
                <h2 style="font-size: 1.8rem; font-weight: 800; color: var(--text-primary);">PROJECT HISTORY</h2>
                <p style="color: var(--text-secondary); font-size: 1rem;">
                    View and manage your previous project risk analyses.
                </p>
            </div>
        """, unsafe_allow_html=True)

    # Fetch real-time user predictions
    predictions = get_user_predictions(user_id)

    with h_top2:
        if predictions:
            if st.button("Clear History", key="btn_clear_all_history", use_container_width=True):
                ok, msg = delete_all_predictions(user_id)
                if ok:
                    st.success(msg)
                    st.session_state.selected_history_project = None
                    st.rerun()
                else:
                    st.error(msg)

    # -------------------------------------------------------------------------
    # EMPTY STATE
    # -------------------------------------------------------------------------
    if not predictions:
        with st.container():
            st.markdown("""
                <div style="background: var(--bg-card); border: 1px solid var(--border-color); border-radius: 18px; padding: 40px; text-align: center; margin-top: 10px; margin-bottom: 24px; box-shadow: var(--card-shadow);">
                    <h3 style="font-size: 1.4rem; font-weight: 800; color: var(--text-primary) !important; margin-bottom: 8px;">No Project Analyses Yet</h3>
                    <p style="color: var(--text-secondary) !important; font-size: 1rem; margin-bottom: 6px;">You have not analyzed any projects yet.</p>
                    <p style="color: var(--text-secondary) !important; opacity: 0.8; font-size: 0.9rem; margin-bottom: 20px;">Your completed project risk analyses will appear here after you perform a risk analysis.</p>
                </div>
            """, unsafe_allow_html=True)

            h_col1, h_col2, h_col3 = st.columns([1.5, 2, 1.5])
            with h_col2:
                if st.button("Start Your First Analysis", key="btn_first_analysis", use_container_width=True):
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
        filtered_preds.sort(key=lambda x: float(x.get("overall_risk_score", x.get("risk_score", 0.0))), reverse=True)
    elif sort_by == "Lowest Risk":
        filtered_preds.sort(key=lambda x: float(x.get("overall_risk_score", x.get("risk_score", 0.0))), reverse=False)

    st.markdown("<br>", unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # ANALYSIS HISTORY TABLE
    # -------------------------------------------------------------------------
    st.markdown("### ANALYSIS HISTORY")

    if not filtered_preds:
        st.info("No matching project analyses found for the selected filters.")
        return

    color_map = {
        "Low": "#10b981",
        "Medium": "#f59e0b",
        "High": "#ef4444",
        "Critical": "#991b1b"
    }

    # Render History Table Headers
    th_col1, th_col2, th_col3, th_col4, th_col5, th_col6, th_col7 = st.columns([2.2, 1.5, 1.2, 1.2, 1.2, 1.5, 1.0])
    with th_col1: st.markdown("**Project Name**")
    with th_col2: st.markdown("**Risk Category**")
    with th_col3: st.markdown("**Confidence**")
    with th_col4: st.markdown("**Overall Score**")
    with th_col5: st.markdown("**Date**")
    with th_col6: st.markdown("**Action**")
    with th_col7: st.markdown("**Delete**")

    st.markdown("<hr style='border-color: var(--border-color); margin: 6px 0 12px 0;'>", unsafe_allow_html=True)

    # Render History Table Rows
    for idx, item in enumerate(filtered_preds):
        p_id = item.get("id") or item.get("_id")
        p_name = item.get("project_name", "Untitled")
        r_level = item.get("risk_level", "Medium")
        p_conf = item.get("prediction_confidence", item.get("risk_score", 0.0))
        o_score = item.get("overall_risk_score", item.get("risk_score", 0.0))
        date_str = item.get("analyzed_at", "N/A")
        badge_c = color_map.get(r_level, "#ef4444")

        r_col1, r_col2, r_col3, r_col4, r_col5, r_col6, r_col7 = st.columns([2.2, 1.5, 1.2, 1.2, 1.2, 1.5, 1.0], vertical_alignment="center")

        with r_col1:
            st.markdown(f"**{p_name}**")
        with r_col2:
            st.markdown(f"<span style='background: {badge_c}; color: #ffffff; font-weight: 700; padding: 3px 10px; border-radius: 10px; font-size: 0.8rem;'>{r_level}</span>", unsafe_allow_html=True)
        with r_col3:
            st.markdown(f"**{p_conf}%**")
        with r_col4:
            st.markdown(f"**{o_score}%**")
        with r_col5:
            st.markdown(f"<span style='color: var(--text-secondary); font-size: 0.82rem;'>{date_str}</span>", unsafe_allow_html=True)
        with r_col6:
            if st.button("View Details", key=f"btn_details_{idx}_{p_id}", use_container_width=True):
                st.session_state.selected_history_project = item
        with r_col7:
            if st.button("Delete", key=f"btn_del_{idx}_{p_id}", use_container_width=True):
                ok, msg = delete_prediction(p_id)
                if ok:
                    st.success(msg)
                    st.session_state.selected_history_project = None
                    st.rerun()
                else:
                    st.error(msg)

        st.markdown("<hr style='border-color: var(--border-color); margin: 6px 0;'>", unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # PROJECT DETAILS MODAL CARD WITH ALL 20 INPUT FEATURES
    # -------------------------------------------------------------------------
    if "selected_history_project" in st.session_state and st.session_state.selected_history_project:
        selected_p = st.session_state.selected_history_project
        s_name = selected_p.get("project_name", "Untitled")
        s_features = selected_p.get("input_features", {})
        s_level = selected_p.get("risk_level", "Medium")
        s_conf = selected_p.get("prediction_confidence", selected_p.get("risk_score", 0.0))
        s_overall = selected_p.get("overall_risk_score", selected_p.get("risk_score", 0.0))
        s_date = selected_p.get("analyzed_at", "N/A")
        badge_color = color_map.get(s_level, "#ef4444")

        st.markdown("<br><br>", unsafe_allow_html=True)
        st.markdown("### PROJECT DETAILS")

        st.markdown(f"""
            <div style="background: var(--bg-card); border: 2px solid {badge_color}; border-radius: 16px; padding: 28px; box-shadow: var(--card-shadow);">
                <div style="font-size: 1.1rem; color: var(--text-primary); margin-bottom: 10px;">
                    <strong>Project Name:</strong> {s_name}
                </div>
                <div style="font-size: 1.1rem; color: var(--text-primary); margin-bottom: 10px; display: flex; align-items: center; gap: 10px;">
                    <strong>Predicted Risk Category:</strong>
                    <span style="background: {badge_color}; color: #ffffff; font-weight: 800; padding: 4px 14px; border-radius: 12px; font-size: 0.9rem;">
                        {s_level.upper()} RISK
                    </span>
                </div>
                <div style="font-size: 1.05rem; color: var(--text-primary); margin-bottom: 8px;">
                    <strong>Prediction Confidence:</strong> <span style="color: {badge_color}; font-weight: 800;">{s_conf}%</span>
                </div>
                <div style="font-size: 1.05rem; color: var(--text-primary); margin-bottom: 16px;">
                    <strong>Overall Risk Score:</strong> <span style="color: var(--text-primary); font-weight: 800;">{s_overall}%</span>
                </div>
                <div style="font-size: 0.88rem; color: var(--text-secondary); margin-bottom: 20px;">
                    <strong>Analyzed On:</strong> {s_date}
                </div>
            </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("#### ALL 20 INPUT FEATURES USED FOR THIS PREDICTION")

        feat_cols1, feat_cols2 = st.columns(2)
        with feat_cols1:
            st.write("**Categorical & Demographics:**")
            st.json({
                "project_type": s_features.get("project_type", "N/A"),
                "industry_sector": s_features.get("industry_sector", "N/A"),
                "methodology": s_features.get("methodology", "N/A"),
                "region": s_features.get("region", "N/A"),
                "priority": s_features.get("priority", "N/A")
            })

            st.write("**Planning & Cost Metrics:**")
            st.json({
                "planned_duration_days": s_features.get("planned_duration_days", "N/A"),
                "budget_usd": s_features.get("budget_usd", "N/A"),
                "requirement_changes_count": s_features.get("requirement_changes_count", "N/A"),
                "vendor_dependency_count": s_features.get("vendor_dependency_count", "N/A"),
                "milestones_missed": s_features.get("milestones_missed", "N/A")
            })

        with feat_cols2:
            st.write("**Team & Resource Metrics:**")
            st.json({
                "team_size": s_features.get("team_size", "N/A"),
                "team_avg_experience_years": s_features.get("team_avg_experience_years", "N/A"),
                "team_turnover_pct": s_features.get("team_turnover_pct", "N/A"),
                "resource_availability_pct": s_features.get("resource_availability_pct", "N/A")
            })

            st.write("**Governance & Quality Scores:**")
            st.json({
                "communication_score": s_features.get("communication_score", "N/A"),
                "sponsor_engagement_score": s_features.get("sponsor_engagement_score", "N/A"),
                "tech_complexity_score": s_features.get("tech_complexity_score", "N/A"),
                "scope_clarity_score": s_features.get("scope_clarity_score", "N/A"),
                "external_dependency_score": s_features.get("external_dependency_score", "N/A"),
                "defect_count": s_features.get("defect_count", "N/A")
            })

        st.markdown("<br>", unsafe_allow_html=True)
        d_col1, d_col2, d_col3 = st.columns([1.5, 2, 1.5])
        with d_col2:
            if st.button("View Visualization", key="btn_hist_viz", use_container_width=True):
                st.session_state.active_tab = "visualization"
                st.rerun()
