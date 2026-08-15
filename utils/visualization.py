"""
Visualization Component for AI-Based Project Risk Forecasting System
Renders detailed project factor charts, risk overview gauge, team & planning metrics, and quality insights.
"""

import streamlit as st
import pandas as pd
from utils.database_client import get_user_predictions


def render_visualization_page():
    """Renders the Project Analysis & Visualization page according to exact user specification."""
    user = st.session_state.get("user_doc", {})
    user_id = user.get("_id", "")

    # Main Header
    st.markdown("""
        <div style="margin-bottom: 24px;">
            <h2 style="font-size: 1.8rem; font-weight: 800; color: #0f172a;">PROJECT ANALYSIS & VISUALIZATION</h2>
            <p style="color: #64748b; font-size: 1rem;">
                Visualize your project data and understand the factors that influence its overall risk.
            </p>
        </div>
    """, unsafe_allow_html=True)

    # Fetch real-time user prediction documents from MongoDB Atlas
    predictions = get_user_predictions(user_id)

    if not predictions:
        st.info("📊 **No project analysis data available yet.**\n\nPlease complete a project risk analysis first to see your project risk visualization.")
        v_col1, v_col2, v_col3 = st.columns([1.5, 2, 1.5])
        with v_col2:
            if st.button("🚀 Start Risk Analysis", key="btn_viz_start"):
                st.session_state.active_tab = "risk_analysis"
                st.rerun()
        return

    # Select project to visualize (default: most recent analyzed project)
    project_names = [p.get("project_name", f"Project {i+1}") for i, p in enumerate(predictions)]

    selected_project_name = st.selectbox("Select Analyzed Project to View Visualization:", project_names, index=0)

    # Find target prediction doc
    target_pred = next((p for p in predictions if p.get("project_name") == selected_project_name), predictions[0])

    project_name = target_pred.get("project_name", "Untitled Project")
    risk_level = target_pred.get("risk_level", "Medium")
    risk_score = target_pred.get("risk_score", 0.0)
    features = target_pred.get("input_features", {})

    project_type = features.get("project_type", "Software Development")

    # Color badge map
    color_map = {
        "Low": "#10b981",
        "Medium": "#f59e0b",
        "High": "#ef4444",
        "Critical": "#dc2626"
    }
    badge_color = color_map.get(risk_level, "#ef4444")

    # -------------------------------------------------------------------------
    # 1. PROJECT SUMMARY
    # -------------------------------------------------------------------------
    st.markdown("### PROJECT SUMMARY")
    st.markdown(f"""
        <div style="background: #ffffff; border: 1px solid #e2e8f0; border-radius: 16px; padding: 24px; margin-bottom: 28px; box-shadow: 0 4px 15px rgba(0,0,0,0.03);">
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 16px;">
                <div>
                    <div style="font-size: 0.85rem; color: #64748b; font-weight: 600;">Project Name</div>
                    <div style="font-size: 1.25rem; font-weight: 800; color: #0f172a;">{project_name}</div>
                </div>
                <div>
                    <div style="font-size: 0.85rem; color: #64748b; font-weight: 600;">Project Type</div>
                    <div style="font-size: 1.15rem; font-weight: 700; color: #334155;">{project_type}</div>
                </div>
                <div>
                    <div style="font-size: 0.85rem; color: #64748b; font-weight: 600;">Predicted Risk</div>
                    <div style="font-size: 1.15rem; font-weight: 800; color: {badge_color};">{risk_level} Risk</div>
                </div>
                <div>
                    <div style="font-size: 0.85rem; color: #64748b; font-weight: 600;">Risk Score</div>
                    <div style="font-size: 1.3rem; font-weight: 800; color: {badge_color};">{risk_score}%</div>
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # 2. RISK OVERVIEW
    # -------------------------------------------------------------------------
    st.markdown("### RISK OVERVIEW")
    st.markdown("<p style='color: #64748b; margin-bottom: 16px;'>A clear visual representation of the predicted project risk.</p>", unsafe_allow_html=True)

    st.markdown(f"""
        <div style="background: #ffffff; border: 1px solid #e2e8f0; border-radius: 16px; padding: 24px; margin-bottom: 32px; box-shadow: 0 4px 15px rgba(0,0,0,0.03);">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                <span style="font-weight: 700; color: #0f172a;">Risk Score Indicator</span>
                <span style="font-weight: 800; color: {badge_color}; font-size: 1.2rem;">{risk_score}%</span>
            </div>
            <div style="display: flex; height: 18px; border-radius: 10px; overflow: hidden; background: #e2e8f0; margin-bottom: 16px;">
                <div style="width: {risk_score}%; background: {badge_color};" title="Risk Score: {risk_score}%"></div>
            </div>
            <div style="display: flex; justify-content: space-between; font-size: 0.88rem; font-weight: 600;">
                <span style="color: #10b981;">🟢 Low Risk (0-35%)</span>
                <span style="color: #f59e0b;">🟡 Medium Risk (36-65%)</span>
                <span style="color: #ef4444;">🔴 High Risk (66-100%)</span>
            </div>
        </div>
    """, unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # 3. PROJECT FACTORS ANALYSIS (BAR CHART)
    # -------------------------------------------------------------------------
    st.markdown("### PROJECT FACTORS ANALYSIS")
    st.markdown("<p style='color: #64748b; margin-bottom: 16px;'>Visualize the key project factors provided during the analysis.</p>", unsafe_allow_html=True)

    factor_data = {
        "Project Factor": [
            "Technical Complexity",
            "Scope Clarity",
            "Communication Score",
            "Sponsor Engagement",
            "External Dependency"
        ],
        "Score": [
            float(features.get("tech_complexity_score", 65.0)),
            float(features.get("scope_clarity_score", 70.0)),
            float(features.get("communication_score", 75.0)),
            float(features.get("sponsor_engagement_score", 80.0)),
            float(features.get("external_dependency_score", 35.0))
        ]
    }
    df_factors = pd.DataFrame(factor_data).set_index("Project Factor")
    st.bar_chart(df_factors, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # 4. TEAM & RESOURCE OVERVIEW
    # -------------------------------------------------------------------------
    st.markdown("### TEAM & RESOURCE OVERVIEW")
    st.markdown("<p style='color: #64748b; margin-bottom: 16px;'>Visualize the team and resource-related project data.</p>", unsafe_allow_html=True)

    t1, t2, t3, t4 = st.columns(4)
    with t1:
        st.metric("Team Size", f"{features.get('team_size', 12)} members")
    with t2:
        st.metric("Average Team Experience", f"{features.get('team_avg_experience_years', 5.5)} Yrs")
    with t3:
        st.metric("Team Turnover", f"{features.get('team_turnover_pct', 10.0)}%")
    with t4:
        st.metric("Resource Availability", f"{features.get('resource_availability_pct', 85.0)}%")

    st.markdown("<br>", unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # 5. PROJECT PLANNING OVERVIEW
    # -------------------------------------------------------------------------
    st.markdown("### PROJECT PLANNING OVERVIEW")
    st.markdown("<p style='color: #64748b; margin-bottom: 16px;'>Visualize important planning and project metrics.</p>", unsafe_allow_html=True)

    p1, p2, p3, p4, p5 = st.columns(5)
    with p1:
        st.metric("Planned Duration", f"{features.get('planned_duration_days', 180)} Days")
    with p2:
        st.metric("Budget", f"${features.get('budget_usd', 250000):,.0f}")
    with p3:
        st.metric("Requirement Changes", f"{features.get('requirement_changes_count', 4)}")
    with p4:
        st.metric("Milestones Missed", f"{features.get('milestones_missed', 1)}")
    with p5:
        st.metric("Vendor Dependency Count", f"{features.get('vendor_dependency_count', 2)}")

    st.markdown("<br>", unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # 6. PROJECT QUALITY OVERVIEW
    # -------------------------------------------------------------------------
    st.markdown("### PROJECT QUALITY OVERVIEW")
    st.markdown("<p style='color: #64748b; margin-bottom: 16px;'>Display the project quality-related information based on the analyzed input.</p>", unsafe_allow_html=True)

    q1, q2 = st.columns([1, 3])
    with q1:
        st.metric("Defect Count", f"{features.get('defect_count', 5)} Defects")

    st.markdown("<br>", unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # 7. PROJECT ANALYSIS SUMMARY
    # -------------------------------------------------------------------------
    st.markdown("""
        <div style="background: #f8fafc; border: 1px solid #cbd5e1; border-left: 5px solid #4f46e5; border-radius: 12px; padding: 24px; margin-top: 12px;">
            <h4 style="font-size: 1.1rem; font-weight: 700; color: #0f172a; margin-bottom: 8px;">PROJECT ANALYSIS SUMMARY</h4>
            <p style="font-size: 0.98rem; color: #475569; margin-bottom: 8px; line-height: 1.5;">
                The visualization above provides an overview of the project factors used in the risk analysis.
            </p>
            <p style="font-size: 0.98rem; color: #475569; line-height: 1.5;">
                Review the project metrics to understand the areas that may require attention and their relationship to the predicted project risk.
            </p>
        </div>
    """, unsafe_allow_html=True)
