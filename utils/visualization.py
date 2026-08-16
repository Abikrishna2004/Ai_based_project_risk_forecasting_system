"""
Visualization Component for AI-Based Project Risk Forecasting System
Renders detailed project factor charts, risk overview gauge using overall_risk_score,
team & planning metrics, and quality insights from saved database prediction records.
"""

import streamlit as st
import pandas as pd
from utils.api_client import get_user_predictions


def render_visualization_page():
    """Renders the Project Analysis & Visualization page using saved database records."""
    user = st.session_state.get("user_doc", {})
    user_id = user.get("_id", "")

    # Main Header
    st.markdown("""
        <div style="margin-bottom: 24px;">
            <h2 style="font-size: 1.8rem; font-weight: 800; color: var(--text-primary);">PROJECT ANALYSIS & VISUALIZATION</h2>
            <p style="color: var(--text-secondary); font-size: 1rem;">
                Visualize your stored project data and understand the factors that influence its overall risk score.
            </p>
        </div>
    """, unsafe_allow_html=True)

    # Fetch real-time user prediction documents from database
    predictions = get_user_predictions(user_id)

    if not predictions:
        st.info("No project analysis data available yet. Please complete a project risk analysis first to see your project risk visualization.")
        v_col1, v_col2, v_col3 = st.columns([1.5, 2, 1.5])
        with v_col2:
            if st.button("Start Risk Analysis", key="btn_viz_start", use_container_width=True):
                st.session_state.active_tab = "risk_analysis"
                st.rerun()
        return

    # Select project to visualize
    project_names = [p.get("project_name", f"Project {i+1}") for i, p in enumerate(predictions)]
    default_idx = 0

    if "selected_history_project" in st.session_state and st.session_state.selected_history_project:
        sel_name = st.session_state.selected_history_project.get("project_name")
        if sel_name in project_names:
            default_idx = project_names.index(sel_name)

    selected_project_name = st.selectbox("Select Analyzed Project to View Visualization:", project_names, index=default_idx)

    # Find target stored prediction record
    target_pred = next((p for p in predictions if p.get("project_name") == selected_project_name), predictions[0])

    project_name = target_pred.get("project_name", "Untitled Project")
    model_predicted_category = target_pred.get("model_predicted_category", target_pred.get("risk_level", "Medium"))
    risk_category = target_pred.get("risk_category", target_pred.get("risk_level", "Medium"))
    prediction_confidence = float(target_pred.get("prediction_confidence", target_pred.get("risk_score", 0.0)))
    overall_risk_score = float(target_pred.get("overall_risk_score", target_pred.get("risk_score", 0.0)))
    features = target_pred.get("input_features", {})

    project_type = features.get("project_type", "Software Development")

    # Clean Enterprise Color Map
    color_map = {
        "Low": "#10b981",       # Green
        "Medium": "#f59e0b",    # Amber/Orange
        "High": "#ef4444",      # Red
        "Critical": "#991b1b"   # Dark Red
    }
    badge_color = color_map.get(risk_category, "#ef4444")
    model_badge_color = color_map.get(model_predicted_category, "#f59e0b")

    # -------------------------------------------------------------------------
    # 1. PROJECT SUMMARY
    # -------------------------------------------------------------------------
    st.markdown("### PROJECT SUMMARY")
    st.markdown(f"""
        <div style="background: var(--bg-card); border: 1px solid var(--border-color); border-radius: 16px; padding: 24px; margin-bottom: 28px; box-shadow: var(--card-shadow);">
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 16px;">
                <div>
                    <div style="font-size: 0.85rem; color: var(--text-secondary); font-weight: 600;">Project Name</div>
                    <div style="font-size: 1.25rem; font-weight: 800; color: var(--text-primary);">{project_name}</div>
                </div>
                <div>
                    <div style="font-size: 0.85rem; color: var(--text-secondary); font-weight: 600;">Project Type</div>
                    <div style="font-size: 1.15rem; font-weight: 700; color: var(--text-primary);">{project_type}</div>
                </div>
                <div>
                    <div style="font-size: 0.85rem; color: var(--text-secondary); font-weight: 600;">Model Prediction</div>
                    <div style="font-size: 1.15rem; font-weight: 800; color: {model_badge_color};">{model_predicted_category.upper()} RISK</div>
                </div>
                <div>
                    <div style="font-size: 0.85rem; color: var(--text-secondary); font-weight: 600;">Overall Risk Category</div>
                    <div style="font-size: 1.15rem; font-weight: 800; color: {badge_color};">{risk_category.upper()} RISK</div>
                </div>
                <div>
                    <div style="font-size: 0.85rem; color: var(--text-secondary); font-weight: 600;">Overall Risk Score</div>
                    <div style="font-size: 1.3rem; font-weight: 800; color: var(--text-primary);">{overall_risk_score}%</div>
                </div>
                <div>
                    <div style="font-size: 0.85rem; color: var(--text-secondary); font-weight: 600;">Prediction Confidence</div>
                    <div style="font-size: 1.2rem; font-weight: 800; color: {badge_color};">{prediction_confidence}%</div>
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # 2. RISK OVERVIEW GAUGE (INDICATOR USES EXACT OVERALL_RISK_SCORE)
    # -------------------------------------------------------------------------
    st.markdown("### RISK OVERVIEW")
    st.markdown("<p style='color: var(--text-secondary); margin-bottom: 16px;'>Visual representation of overall project risk severity score.</p>", unsafe_allow_html=True)

    st.markdown(f"""
        <div style="background: var(--bg-card); border: 1px solid var(--border-color); border-radius: 16px; padding: 24px; margin-bottom: 32px; box-shadow: var(--card-shadow);">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                <span style="font-weight: 700; color: var(--text-primary);">Overall Risk Severity Indicator</span>
                <span style="font-weight: 800; color: {badge_color}; font-size: 1.25rem;">Overall Risk Score: {overall_risk_score}%</span>
            </div>
            <div style="display: flex; height: 20px; border-radius: 10px; overflow: hidden; background: rgba(225, 29, 126, 0.1); margin-bottom: 16px;">
                <div style="width: {overall_risk_score}%; background: {badge_color};" title="Overall Risk Score: {overall_risk_score}%"></div>
            </div>
            <div style="display: flex; justify-content: space-between; font-size: 0.88rem; font-weight: 600;">
                <span style="color: #10b981;">Low Risk (0–35%)</span>
                <span style="color: #f59e0b;">Medium Risk (36–65%)</span>
                <span style="color: #ef4444;">High Risk (66–85%)</span>
                <span style="color: #991b1b;">Critical Risk (86–100%)</span>
            </div>
            <div style="margin-top: 14px; padding-top: 12px; border-top: 1px dashed var(--border-color); font-size: 0.9rem; color: var(--text-secondary);">
                <strong>Prediction Confidence:</strong> {prediction_confidence}% (Probability of model's highest predicted class '{model_predicted_category}')
            </div>
        </div>
    """, unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # 3. PROJECT FACTORS ANALYSIS (BAR CHART FOR 5 GOVERNANCE SCORES)
    # -------------------------------------------------------------------------
    st.markdown("### PROJECT FACTORS ANALYSIS")
    st.markdown("<p style='color: var(--text-secondary); margin-bottom: 16px;'>Key governance and assessment scores provided during analysis.</p>", unsafe_allow_html=True)

    factor_data = {
        "Project Factor": [
            "Communication Score",
            "Sponsor Engagement Score",
            "Technical Complexity Score",
            "Scope Clarity Score",
            "External Dependency Score"
        ],
        "Score": [
            float(features.get("communication_score", 75.0)),
            float(features.get("sponsor_engagement_score", 80.0)),
            float(features.get("tech_complexity_score", 65.0)),
            float(features.get("scope_clarity_score", 70.0)),
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
    st.markdown("<p style='color: var(--text-secondary); margin-bottom: 16px;'>Team composition and resource allocation metrics.</p>", unsafe_allow_html=True)

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
    st.markdown("<p style='color: var(--text-secondary); margin-bottom: 16px;'>Schedule, budget, and planning metrics.</p>", unsafe_allow_html=True)

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
    st.markdown("<p style='color: var(--text-secondary); margin-bottom: 16px;'>Quality defects and error metrics.</p>", unsafe_allow_html=True)

    q1, q2 = st.columns([1, 3])
    with q1:
        st.metric("Defect Count", f"{features.get('defect_count', 5)} Defects")

    st.markdown("<br>", unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # 7. PROJECT ANALYSIS SUMMARY
    # -------------------------------------------------------------------------
    st.markdown("""
        <div style="background: var(--bg-card); border: 1px solid var(--border-color); border-left: 5px solid #e11d74; border-radius: 12px; padding: 24px; margin-top: 12px; box-shadow: var(--card-shadow);">
            <h4 style="font-size: 1.1rem; font-weight: 700; color: var(--text-primary); margin-bottom: 8px;">PROJECT ANALYSIS SUMMARY</h4>
            <p style="font-size: 0.98rem; color: var(--text-secondary); margin-bottom: 8px; line-height: 1.5;">
                The visualization above displays the stored 20 input features and overall risk score retrieved from the database.
            </p>
            <p style="font-size: 0.98rem; color: var(--text-secondary); line-height: 1.5;">
                Review the project metrics to understand key risk drivers and capacity constraints.
            </p>
        </div>
    """, unsafe_allow_html=True)
