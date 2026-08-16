"""
Project Risk Analysis Form, Document Upload & Result Component
Supports Manual Form Entry, CSV Spreadsheet Upload, and PDF Document Text Parsing.
Performs 20-feature extraction, normalization, review preview, and CatBoost risk forecasting.
"""

import streamlit as st
from utils.predictor import predict_project_risk
from utils.api_client import save_project_prediction
from utils.project_data_extractor import (
    extract_from_csv,
    extract_from_pdf,
    validate_extracted_features,
    VALID_CATEGORICAL_OPTIONS,
    REQUIRED_ML_FEATURES
)


def render_risk_analysis_page():
    """Renders the Project Risk Analysis page with manual entry, CSV, and PDF document upload options."""
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
                Enter project details manually or upload a CSV / PDF project document to generate real-time AI risk forecasts using CatBoost.
            </p>
        </div>
    """, unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # INPUT METHOD SELECTION
    # -------------------------------------------------------------------------
    input_method = st.radio(
        "Choose Project Data Input Method:",
        ["Manual Form Entry", "Upload CSV File", "Upload PDF Document"],
        horizontal=True,
        key="radio_input_method"
    )

    st.markdown("<br>", unsafe_allow_html=True)

    extracted_dict = {}
    input_source = "manual"
    csv_records = []

    # -------------------------------------------------------------------------
    # OPTION 2: CSV FILE UPLOAD
    # -------------------------------------------------------------------------
    if input_method == "Upload CSV File":
        input_source = "csv"
        uploaded_csv = st.file_uploader("Upload Project Data CSV File", type=["csv"], key="uploader_csv")

        if uploaded_csv is not None:
            records, err = extract_from_csv(uploaded_csv)
            if err:
                st.error(err)
            elif not records:
                st.warning("No records could be extracted from the uploaded CSV.")
            else:
                csv_records = records
                st.success(f"Successfully extracted {len(records)} project record(s) from CSV!")

                if len(records) > 1:
                    row_options = [
                        f"Row {i+1}: {r.get('project_name') or 'Project ' + str(i+1)}"
                        for i, r in enumerate(records)
                    ]
                    selected_idx = st.selectbox("Select Project Row to Analyze:", range(len(row_options)), format_func=lambda i: row_options[i])
                    extracted_dict = records[selected_idx]
                else:
                    extracted_dict = records[0]

    # -------------------------------------------------------------------------
    # OPTION 3: PDF DOCUMENT UPLOAD
    # -------------------------------------------------------------------------
    elif input_method == "Upload PDF Document":
        input_source = "pdf"
        uploaded_pdf = st.file_uploader("Upload Project Document PDF File", type=["pdf"], key="uploader_pdf")

        if uploaded_pdf is not None:
            extracted, err = extract_from_pdf(uploaded_pdf)
            if err:
                st.error(err)
            else:
                extracted_dict = extracted
                st.success("Successfully extracted project text and feature parameters from PDF document!")

    # Check missing fields for Uploaded CSV / PDF
    missing_fields, cleaned_features = validate_extracted_features(extracted_dict) if extracted_dict else (REQUIRED_ML_FEATURES, {})

    if input_method in ["Upload CSV File", "Upload PDF Document"] and extracted_dict:
        if missing_fields:
            st.warning("Some required project information could not be automatically extracted from the uploaded document.")
            st.markdown(f"**Missing features ({len(missing_fields)}):**")
            for mf in missing_fields:
                st.markdown(f"- `{mf}`")
            st.info("Please complete the missing field values in the form below before proceeding with the risk analysis.")
        else:
            st.markdown("### EXTRACTED PROJECT INFORMATION PREVIEW")
            st.markdown("Review and confirm the extracted values before analyzing project risk.")

    # Default baseline values for form controls
    def _get_cat_idx(field, default_val):
        val = extracted_dict.get(field) or default_val
        opts = VALID_CATEGORICAL_OPTIONS.get(field, [])
        return opts.index(val) if val in opts else 0

    def _get_num_val(field, default_val, is_int=False):
        v = extracted_dict.get(field)
        if v is None:
            return int(default_val) if is_int else float(default_val)
        try:
            return int(v) if is_int else float(v)
        except (ValueError, TypeError):
            return int(default_val) if is_int else float(default_val)

    # -------------------------------------------------------------------------
    # MAIN FORM CONTAINER (SERVES MANUAL ENTRY & REVIEW FOR EXTRACTED DATA)
    # -------------------------------------------------------------------------
    with st.form("risk_analysis_form"):
        # SECTION 1: PROJECT PLANNING & OVERVIEW
        st.markdown("<h4 style='font-size: 1.15rem; font-weight: 800; color: var(--text-primary); margin-bottom: 14px;'>SECTION 1: PROJECT PLANNING & OVERVIEW</h4>", unsafe_allow_html=True)

        f_col1, f_col2, f_col3 = st.columns(3)

        with f_col1:
            project_name = st.text_input(
                "Project Name",
                value=str(extracted_dict.get("project_name") or "Global Enterprise AI Platform"),
                help="Identifier for tracking"
            )
            project_type = st.selectbox(
                "Project Type",
                VALID_CATEGORICAL_OPTIONS["project_type"],
                index=_get_cat_idx("project_type", "Software Development")
            )
            industry_sector = st.selectbox(
                "Industry Sector",
                VALID_CATEGORICAL_OPTIONS["industry_sector"],
                index=_get_cat_idx("industry_sector", "IT")
            )

        with f_col2:
            methodology = st.selectbox(
                "Development Methodology",
                VALID_CATEGORICAL_OPTIONS["methodology"],
                index=_get_cat_idx("methodology", "Agile")
            )
            region = st.selectbox(
                "Operating Region",
                VALID_CATEGORICAL_OPTIONS["region"],
                index=_get_cat_idx("region", "Asia Pacific")
            )
            priority = st.selectbox(
                "Project Priority",
                VALID_CATEGORICAL_OPTIONS["priority"],
                index=_get_cat_idx("priority", "High")
            )

        with f_col3:
            planned_duration_days = st.number_input(
                "Planned Duration (Days)",
                min_value=1, max_value=3650,
                value=_get_num_val("planned_duration_days", 180, is_int=True),
                step=10
            )
            budget_usd = st.number_input(
                "Budget (USD)",
                min_value=1000, max_value=100000000,
                value=_get_num_val("budget_usd", 250000, is_int=True),
                step=10000
            )
            requirement_changes_count = st.number_input(
                "Requirement Changes Count",
                min_value=0, max_value=100,
                value=_get_num_val("requirement_changes_count", 4, is_int=True),
                step=1
            )

        st.markdown("<hr style='border-color: var(--border-color); margin: 22px 0;'>", unsafe_allow_html=True)

        # SECTION 2: TEAM & RESOURCE CAPACITY
        st.markdown("<h4 style='font-size: 1.15rem; font-weight: 800; color: var(--text-primary); margin-bottom: 14px;'>SECTION 2: TEAM & RESOURCE CAPACITY</h4>", unsafe_allow_html=True)

        t_col1, t_col2, t_col3 = st.columns(3)

        with t_col1:
            team_size = st.number_input(
                "Team Size",
                min_value=1, max_value=500,
                value=_get_num_val("team_size", 12, is_int=True),
                step=1
            )
            team_avg_experience_years = st.number_input(
                "Avg Experience (Years)",
                min_value=0.0, max_value=40.0,
                value=_get_num_val("team_avg_experience_years", 5.5),
                step=0.5
            )

        with t_col2:
            team_turnover_pct = st.number_input(
                "Team Turnover Rate (%)",
                min_value=0.0, max_value=100.0,
                value=_get_num_val("team_turnover_pct", 10.0),
                step=1.0
            )
            resource_availability_pct = st.number_input(
                "Resource Availability (%)",
                min_value=0.0, max_value=100.0,
                value=_get_num_val("resource_availability_pct", 85.0),
                step=5.0
            )

        with t_col3:
            vendor_dependency_count = st.number_input(
                "Vendor Dependency Count",
                min_value=0, max_value=50,
                value=_get_num_val("vendor_dependency_count", 2, is_int=True),
                step=1
            )
            milestones_missed = st.number_input(
                "Milestones Missed",
                min_value=0, max_value=50,
                value=_get_num_val("milestones_missed", 1, is_int=True),
                step=1
            )

        st.markdown("<hr style='border-color: var(--border-color); margin: 22px 0;'>", unsafe_allow_html=True)

        # SECTION 3: GOVERNANCE & QUALITY SCORES
        st.markdown("<h4 style='font-size: 1.15rem; font-weight: 800; color: var(--text-primary); margin-bottom: 14px;'>SECTION 3: GOVERNANCE & QUALITY SCORES</h4>", unsafe_allow_html=True)

        a_col1, a_col2, a_col3 = st.columns(3)

        with a_col1:
            communication_score = st.slider("Communication Score", 0.0, 100.0, _get_num_val("communication_score", 75.0), 1.0)
            sponsor_engagement_score = st.slider("Sponsor Engagement Score", 0.0, 100.0, _get_num_val("sponsor_engagement_score", 80.0), 1.0)

        with a_col2:
            tech_complexity_score = st.slider("Technical Complexity Score", 0.0, 100.0, _get_num_val("tech_complexity_score", 65.0), 1.0)
            scope_clarity_score = st.slider("Scope Clarity Score", 0.0, 100.0, _get_num_val("scope_clarity_score", 70.0), 1.0)

        with a_col3:
            external_dependency_score = st.slider("External Dependency Score", 0.0, 100.0, _get_num_val("external_dependency_score", 35.0), 1.0)
            defect_count = st.number_input("Defect Count", min_value=0, max_value=500, value=_get_num_val("defect_count", 5, is_int=True), step=1)

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
            "planned_duration_days": float(planned_duration_days),
            "budget_usd": float(budget_usd),
            "requirement_changes_count": float(requirement_changes_count),
            "vendor_dependency_count": float(vendor_dependency_count),
            "milestones_missed": float(milestones_missed),
            "team_size": float(team_size),
            "team_avg_experience_years": float(team_avg_experience_years),
            "team_turnover_pct": float(team_turnover_pct),
            "resource_availability_pct": float(resource_availability_pct),
            "communication_score": float(communication_score),
            "sponsor_engagement_score": float(sponsor_engagement_score),
            "tech_complexity_score": float(tech_complexity_score),
            "scope_clarity_score": float(scope_clarity_score),
            "external_dependency_score": float(external_dependency_score),
            "defect_count": float(defect_count)
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

                # Save prediction result to database with input_source metadata
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
                    class_probabilities=class_probs,
                    input_source=input_source
                )

                if not save_ok:
                    st.warning(f"Note: {save_msg}")

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
                            <strong>Data Source:</strong>
                            <span style="background: rgba(225, 29, 126, 0.2); color: #fb7185; font-weight: 800; padding: 4px 14px; border-radius: 12px; font-size: 0.88rem; text-transform: uppercase;">
                                {input_source}
                            </span>
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
