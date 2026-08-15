"""
Risk Analysis Module for AI-Based Project Risk Forecasting System
Implements exact form fields, 20-feature CatBoost model inference, database storage, and result card layout.
"""

import streamlit as st
from utils.predictor import predict_project_risk
from utils.database_client import save_project_prediction


def render_risk_analysis_page():
    """Renders the Project Risk Analysis form and prediction result interface."""
    user = st.session_state.get("user_doc", {})
    user_id = user.get("_id", "")
    email = user.get("email", "")

    st.markdown("""
        <div style="margin-bottom: 24px;">
            <h2 style="font-size: 1.8rem; font-weight: 800; color: #0f172a;">PROJECT RISK ANALYSIS</h2>
            <p style="color: #64748b; font-size: 1rem;">
                Enter your project information below to analyze potential project risk using the trained machine learning model.
            </p>
        </div>
    """, unsafe_allow_html=True)

    with st.form("risk_analysis_form"):

        # ---------------------------------------------------------------------
        # 1. PROJECT INFORMATION
        # ---------------------------------------------------------------------
        st.markdown("### PROJECT INFORMATION")
        col1, col2, col3 = st.columns(3)
        with col1:
            project_name = st.text_input("Project Name *", placeholder="Enter project name")
            project_type = st.selectbox(
                "Project Type *",
                ["Software Development", "IT Infrastructure", "ERP Implementation", "Healthcare IT", "Financial Systems", "Construction", "Manufacturing", "Marketing Campaign", "R&D", "Telecom"]
            )
        with col2:
            industry_sector = st.selectbox(
                "Industry Sector *",
                ["IT", "Finance", "Healthcare", "Manufacturing", "Retail", "Telecom", "Government", "Construction"]
            )
            methodology = st.selectbox("Methodology *", ["Agile", "Hybrid", "Waterfall"])
        with col3:
            region = st.selectbox(
                "Region *",
                ["North America", "Europe", "Asia Pacific", "Latin America", "Middle East", "Africa"]
            )
            priority = st.selectbox("Priority *", ["High", "Medium", "Low", "Critical"])

        st.markdown("<hr style='border-color: #e2e8f0; margin: 20px 0;'>", unsafe_allow_html=True)

        # ---------------------------------------------------------------------
        # 2. PROJECT PLANNING
        # ---------------------------------------------------------------------
        st.markdown("### PROJECT PLANNING")
        c4, c5, c6 = st.columns(3)
        with c4:
            planned_duration_days = st.number_input("Planned Duration (Days) *", min_value=1, max_value=2000, value=180, help="Enter planned duration")
            budget_usd = st.number_input("Budget (USD) *", min_value=1000, max_value=100000000, value=250000, help="Enter project budget")
        with c5:
            requirement_changes_count = st.number_input("Requirement Changes *", min_value=0, max_value=100, value=4, help="Enter number of requirement changes")
            vendor_dependency_count = st.number_input("Vendor Dependency Count *", min_value=0, max_value=20, value=2, help="Enter number of vendor dependencies")
        with c6:
            milestones_missed = st.number_input("Milestones Missed *", min_value=0, max_value=30, value=1, help="Enter number of milestones missed")

        st.markdown("<hr style='border-color: #e2e8f0; margin: 20px 0;'>", unsafe_allow_html=True)

        # ---------------------------------------------------------------------
        # 3. TEAM INFORMATION
        # ---------------------------------------------------------------------
        st.markdown("### TEAM INFORMATION")
        c7, c8, c9, c10 = st.columns(4)
        with c7:
            team_size = st.number_input("Team Size *", min_value=1, max_value=500, value=12, help="Enter team size")
        with c8:
            team_avg_experience_years = st.number_input("Average Team Experience (Years) *", min_value=0.0, max_value=40.0, value=5.5, help="Enter average experience")
        with c9:
            team_turnover_pct = st.number_input("Team Turnover (%) *", min_value=0.0, max_value=100.0, value=10.0, help="Enter turnover percentage")
        with c10:
            resource_availability_pct = st.number_input("Resource Availability (%) *", min_value=0.0, max_value=100.0, value=85.0, help="Enter resource availability")

        st.markdown("<hr style='border-color: #e2e8f0; margin: 20px 0;'>", unsafe_allow_html=True)

        # ---------------------------------------------------------------------
        # 4. PROJECT ASSESSMENT
        # ---------------------------------------------------------------------
        st.markdown("### PROJECT ASSESSMENT")
        c11, c12, c13 = st.columns(3)
        with c11:
            communication_score = st.number_input("Communication Score *", min_value=0.0, max_value=100.0, value=75.0, help="Enter communication score")
            sponsor_engagement_score = st.number_input("Sponsor Engagement Score *", min_value=0.0, max_value=100.0, value=80.0, help="Enter sponsor engagement score")
        with c12:
            tech_complexity_score = st.number_input("Technical Complexity Score *", min_value=0.0, max_value=100.0, value=65.0, help="Enter technical complexity score")
            scope_clarity_score = st.number_input("Scope Clarity Score *", min_value=0.0, max_value=100.0, value=70.0, help="Enter scope clarity score")
        with c13:
            external_dependency_score = st.number_input("External Dependency Score *", min_value=0.0, max_value=100.0, value=35.0, help="Enter external dependency score")
            defect_count = st.number_input("Defect Count *", min_value=0, max_value=200, value=5, help="Enter number of defects")

        st.markdown("<br>", unsafe_allow_html=True)
        btn_col1, btn_col2, btn_col3 = st.columns([1.5, 2, 1.5])
        with btn_col2:
            submit_analysis = st.form_submit_button("Analyze Project", use_container_width=True)

    # -------------------------------------------------------------------------
    # AFTER ANALYSIS RESULT DISPLAY
    # -------------------------------------------------------------------------
    if submit_analysis:
        if not project_name.strip():
            st.error("Please enter a valid Project Name before running the analysis.")
        else:
            # Construct feature payload containing ONLY project_name + exact 20 ML features
            input_dict = {
                "project_name": project_name.strip(),
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
                    st.error(f"❌ Prediction Error: {res['error']}")
                else:
                    risk_level = res["risk_category"]
                    risk_score = res["risk_score"]

                    # Save prediction result to database in real-time
                    save_ok, save_msg = save_project_prediction(
                        user_id=user_id,
                        email=email,
                        project_name=project_name,
                        risk_level=risk_level,
                        risk_score=risk_score,
                        input_features=input_dict
                    )

                if save_ok:
                    st.balloons()

                    # Color badge map
                    color_map = {
                        "Low": "#10b981",
                        "Medium": "#f59e0b",
                        "High": "#ef4444",
                        "Critical": "#dc2626"
                    }
                    badge_color = color_map.get(risk_level, "#ef4444")

                    # PROJECT RISK RESULT CARD
                    st.markdown(f"""
                        <div style="background: #ffffff; border: 2px solid {badge_color}; border-radius: 18px; padding: 32px; margin-top: 28px; box-shadow: 0 10px 30px rgba(0,0,0,0.06);">
                            <h3 style="font-size: 1.4rem; font-weight: 800; color: #0f172a; margin-bottom: 20px; border-bottom: 2px solid #f1f5f9; padding-bottom: 12px;">
                                PROJECT RISK RESULT
                            </h3>
                            <div style="font-size: 1.1rem; color: #334155; margin-bottom: 12px;">
                                <strong>Project Name:</strong> {project_name}
                            </div>
                            <div style="font-size: 1.1rem; color: #334155; margin-bottom: 12px; display: flex; align-items: center; gap: 12px;">
                                <strong>Predicted Risk Category:</strong>
                                <span style="background: {badge_color}; color: #ffffff; font-weight: 800; padding: 6px 20px; border-radius: 20px; font-size: 1.1rem; text-transform: uppercase;">
                                    {risk_level} RISK
                                </span>
                            </div>
                            <div style="font-size: 1.1rem; color: #334155; margin-bottom: 20px;">
                                <strong>Risk Score:</strong> <span style="color: {badge_color}; font-weight: 800; font-size: 1.6rem;">{risk_score}%</span>
                            </div>
                            <div style="background: #f8fafc; border-left: 4px solid #4f46e5; padding: 14px 18px; border-radius: 8px; font-size: 0.95rem; color: #475569; margin-bottom: 20px;">
                                💡 <strong>Analysis Complete:</strong> The project information has been analyzed using the trained CatBoost model. Record saved live in database.
                            </div>
                        </div>
                    """, unsafe_allow_html=True)

                    st.markdown("<br>", unsafe_allow_html=True)
                    r_col1, r_col2, r_col3 = st.columns([1.5, 2, 1.5])
                    with r_col2:
                        if st.button("📊 View Visualization", key="btn_view_viz", use_container_width=True):
                            st.session_state.active_tab = "visualization"
                            st.rerun()
                else:
                    st.error(f"❌ {save_msg}")
