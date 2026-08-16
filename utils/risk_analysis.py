"""
Project Risk Analysis Form & Result Dashboard
Provides 20-feature input form, clean enterprise result cards, and real-time prediction storage.
"""

import streamlit as st
from utils.predictor import predict_project_risk
from utils.api_client import save_project_prediction


def render_risk_analysis_page():
    """Renders the Project Risk Analysis form and prediction results."""
    user = st.session_state.get("user_doc", {})
    user_id = user.get("_id", "")
    email = user.get("email", "")

    # Header Section
    st.markdown("""
        <div style="margin-bottom: 28px;">
            <h2 style="font-size: 1.88rem; font-weight: 800; color: var(--text-primary); margin-bottom: 6px;">
                PROJECT RISK ANALYSIS
            </h2>
            <p style="color: var(--text-secondary); font-size: 1.02rem;">
                Enter project details to generate real-time AI risk forecasts using the CatBoost machine learning model.
            </p>
        </div>
    """, unsafe_allow_html=True)

    with st.form("risk_analysis_form"):
        # ---------------------------------------------------------------------
        # SECTION 1: PROJECT PLANNING & DEMOGRAPHICS
        # ---------------------------------------------------------------------
        st.markdown("<h4 style='font-size: 1.15rem; font-weight: 800; color: var(--text-primary); margin-bottom: 14px;'>SECTION 1: PROJECT PLANNING & OVERVIEW</h4>", unsafe_allow_html=True)

        f_col1, f_col2, f_col3 = st.columns(3)

        with f_col1:
            project_name = st.text_input("Project Name", value="Global Enterprise AI Platform", help="Identifier for tracking")
            project_type = st.selectbox(
                "Project Type",
                ["Software Development", "IT Infrastructure", "ERP Implementation", "Healthcare IT", "Financial Systems", "Construction", "Manufacturing", "Marketing Campaign", "R&D", "Telecom"]
            )
            industry_sector = st.selectbox(
                "Industry Sector",
                ["IT", "Finance", "Healthcare", "Manufacturing", "Retail", "Telecom", "Government", "Construction"]
            )

        with f_col2:
            methodology = st.selectbox("Development Methodology", ["Agile", "Hybrid", "Waterfall"])
            region = st.selectbox("Operating Region", ["North America", "Europe", "Asia Pacific", "Latin America", "Middle East", "Africa"])
            priority = st.selectbox("Project Priority", ["High", "Medium", "Low", "Critical"])

        with f_col3:
            planned_duration_days = st.number_input("Planned Duration (Days)", min_value=1, max_value=3650, value=180, step=10)
            budget_usd = st.number_input("Budget (USD)", min_value=1000, max_value=100000000, value=250000, step=10000)
            requirement_changes_count = st.number_input("Requirement Changes Count", min_value=0, max_value=100, value=4, step=1)

        st.markdown("<hr style='border-color: var(--border-color); margin: 22px 0;'>", unsafe_allow_html=True)

        # ---------------------------------------------------------------------
        # SECTION 2: TEAM & RESOURCE METRICS
        # ---------------------------------------------------------------------
        st.markdown("<h4 style='font-size: 1.15rem; font-weight: 800; color: var(--text-primary); margin-bottom: 14px;'>SECTION 2: TEAM & RESOURCE CAPACITY</h4>", unsafe_allow_html=True)

        t_col1, t_col2, t_col3 = st.columns(3)

        with t_col1:
            team_size = st.number_input("Team Size", min_value=1, max_value=500, value=12, step=1)
            team_avg_experience_years = st.number_input("Avg Experience (Years)", min_value=0.0, max_value=40.0, value=5.5, step=0.5)

        with t_col2:
            team_turnover_pct = st.number_input("Team Turnover Rate (%)", min_value=0.0, max_value=100.0, value=10.0, step=1.0)
            resource_availability_pct = st.number_input("Resource Availability (%)", min_value=0.0, max_value=100.0, value=85.0, step=5.0)

        with t_col3:
            vendor_dependency_count = st.number_input("Vendor Dependency Count", min_value=0, max_value=50, value=2, step=1)
            milestones_missed = st.number_input("Milestones Missed", min_value=0, max_value=50, value=1, step=1)

        st.markdown("<hr style='border-color: var(--border-color); margin: 22px 0;'>", unsafe_allow_html=True)

        # ---------------------------------------------------------------------
        # SECTION 3: PROJECT ASSESSMENT & QUALITY METRICS
        # ---------------------------------------------------------------------
        st.markdown("<h4 style='font-size: 1.15rem; font-weight: 800; color: var(--text-primary); margin-bottom: 14px;'>SECTION 3: GOVERNANCE & QUALITY SCORES</h4>", unsafe_allow_html=True)

        a_col1, a_col2, a_col3 = st.columns(3)

        with a_col1:
            communication_score = st.slider("Communication Score", 0.0, 100.0, 75.0, 1.0)
            sponsor_engagement_score = st.slider("Sponsor Engagement Score", 0.0, 100.0, 80.0, 1.0)

        with a_col2:
            tech_complexity_score = st.slider("Technical Complexity Score", 0.0, 100.0, 65.0, 1.0)
            scope_clarity_score = st.slider("Scope Clarity Score", 0.0, 100.0, 70.0, 1.0)

        with a_col3:
            external_dependency_score = st.slider("External Dependency Score", 0.0, 100.0, 35.0, 1.0)
            defect_count = st.number_input("Defect Count", min_value=0, max_value=500, value=5, step=1)

        st.markdown("<br>", unsafe_allow_html=True)
        btn_submit = st.form_submit_button("Analyze Project Risk", use_container_width=True)

    # -------------------------------------------------------------------------
    # PROCESS PREDICTION & RENDER RESULT CARD
    # -------------------------------------------------------------------------
    if btn_submit:
        if not project_name.strip():
            st.error("Please enter a valid project name before submitting.")
            return

        input_dict = {
            "project_type": project_type,
            "industry_sector": industry_sector,
            "methodology": methodology,
            "region": region,
            "priority": priority,
            "planned_duration_days": planned_duration_days,
            "budget_usd": budget_usd,
            "requirement_changes_count": requirement_changes_count,
            "vendor_dependency_count": vendor_dependency_count,
            "milestones_missed": milestones_missed,
            "team_size": team_size,
            "team_avg_experience_years": team_avg_experience_years,
            "team_turnover_pct": team_turnover_pct,
            "resource_availability_pct": resource_availability_pct,
            "communication_score": communication_score,
            "sponsor_engagement_score": sponsor_engagement_score,
            "tech_complexity_score": tech_complexity_score,
            "scope_clarity_score": scope_clarity_score,
            "external_dependency_score": external_dependency_score,
            "defect_count": defect_count
        }

        with st.spinner("Analyzing project parameters using CatBoost model..."):
            res = predict_project_risk(input_dict)

            if "error" in res:
                st.error(res["error"])
            else:
                model_pred_category = res.get("model_predicted_category", "Medium")
                overall_risk_level = res.get("risk_category", "Medium")
                overall_risk_score = res.get("overall_risk_score", 50.0)
                pred_confidence = res.get("prediction_confidence", 70.0)
                class_probs = res.get("class_probabilities", {})

                # Save prediction result to database
                save_ok, save_msg = save_project_prediction(
                    user_id=user_id,
                    email=email,
                    project_name=project_name,
                    risk_level=overall_risk_level,
                    risk_score=pred_confidence,
                    input_features=input_dict,
                    model_predicted_category=model_pred_category,
                    risk_category=overall_risk_level,
                    overall_risk_score=overall_risk_score,
                    prediction_confidence=pred_confidence,
                    class_probabilities=class_probs
                )

                if save_ok:
                    color_map = {
                        "Low": "#10b981",       # Green
                        "Medium": "#f59e0b",    # Amber/Orange
                        "High": "#ef4444",      # Red
                        "Critical": "#991b1b"   # Dark Red
                    }
                    badge_color = color_map.get(overall_risk_level, "#ef4444")
                    model_badge_color = color_map.get(model_pred_category, "#f59e0b")

                    st.markdown("<br>", unsafe_allow_html=True)
                    st.markdown(f"""
                        <div style="background: var(--bg-card); border: 2px solid {badge_color}; border-radius: 16px; padding: 30px; margin-top: 24px; box-shadow: var(--card-shadow);">
                            <h3 style="font-size: 1.35rem; font-weight: 800; color: var(--text-primary); margin-bottom: 20px; border-bottom: 1px solid var(--border-color); padding-bottom: 12px;">
                                PROJECT RISK RESULT
                            </h3>
                            <div style="margin-bottom: 14px; font-size: 1.05rem; color: var(--text-primary);">
                                <strong>Project Name:</strong> {project_name}
                            </div>
                            <div style="margin-bottom: 14px; font-size: 1.05rem; color: var(--text-primary); display: flex; align-items: center; gap: 12px;">
                                <strong>Model Prediction:</strong>
                                <span style="background: {model_badge_color}; color: #ffffff; font-weight: 800; padding: 4px 14px; border-radius: 12px; font-size: 0.92rem;">
                                    {model_pred_category.upper()} RISK
                                </span>
                            </div>
                            <div style="margin-bottom: 14px; font-size: 1.05rem; color: var(--text-primary); display: flex; align-items: center; gap: 12px;">
                                <strong>Overall Risk Level:</strong>
                                <span style="background: {badge_color}; color: #ffffff; font-weight: 800; padding: 4px 14px; border-radius: 12px; font-size: 0.92rem;">
                                    {overall_risk_level.upper()} RISK
                                </span>
                            </div>
                            <div style="margin-bottom: 14px; font-size: 1.05rem; color: var(--text-primary);">
                                <strong>Overall Risk Score:</strong> <span style="color: var(--text-primary); font-weight: 800; font-size: 1.3rem;">{overall_risk_score}%</span>
                            </div>
                            <div style="margin-bottom: 18px; font-size: 1.05rem; color: var(--text-primary);">
                                <strong>Prediction Confidence:</strong> <span style="color: {badge_color}; font-weight: 800; font-size: 1.25rem;">{pred_confidence}%</span>
                            </div>
                            <div style="font-size: 0.92rem; color: var(--text-secondary); line-height: 1.5; border-top: 1px dashed var(--border-color); padding-top: 14px;">
                                <strong>Analysis Status:</strong> The project information has been successfully analyzed using the trained CatBoost model and saved to the database.
                            </div>
                        </div>
                    """, unsafe_allow_html=True)

                    st.markdown("<br>", unsafe_allow_html=True)
                    r_col1, r_col2, r_col3 = st.columns([1, 2, 1])
                    with r_col2:
                        if st.button("View Visualization", key="btn_view_viz_result", use_container_width=True):
                            st.session_state.active_tab = "visualization"
                            st.rerun()
