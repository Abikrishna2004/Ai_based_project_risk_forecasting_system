"""
AI-Based Project Risk Forecasting System
Main Streamlit Entrypoint with Landing Page, Real-Time Authentication, & Main Dashboard
"""

import os
import streamlit as st
from dotenv import load_dotenv
from utils.landing import render_landing_page
from utils.dashboard import render_main_dashboard, render_top_navigation
from utils.risk_analysis import render_risk_analysis_page
from utils.visualization import render_visualization_page
from utils.history import render_history_page
from utils.profile import render_profile_page
from utils.validators import is_valid_email, validate_registration_fields
from utils.api_client import (
    register_user,
    authenticate_user
)

# Load environment variables from .env
load_dotenv()

# Page Configuration
st.set_page_config(
    page_title="AI-Based Project Risk Forecasting System",
    page_icon="AI",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Inject Master Global Responsive CSS for All Viewports (Desktop, Tablet, Mobile)
st.markdown("""
    <style>
        /* Global Reset & Responsive Box Sizing */
        html, body {
            box-sizing: border-box;
            overflow-x: hidden !important;
        }
        *, *:before, *:after {
            box-sizing: inherit;
        }

        /* Container Max Widths */
        .block-container {
            max-width: 1240px !important;
            padding-left: 1.5rem !important;
            padding-right: 1.5rem !important;
        }

        /* Tablet Responsiveness (768px - 1024px) */
        @media (min-width: 768px) and (max-width: 1024px) {
            .block-container {
                max-width: 95% !important;
                padding-left: 1rem !important;
                padding-right: 1rem !important;
            }
            div[data-testid="stHorizontalBlock"] {
                flex-wrap: wrap !important;
                gap: 12px !important;
            }
            div[data-testid="column"] {
                min-width: 45% !important;
                flex: 1 1 45% !important;
            }
        }

        /* Mobile Responsiveness (< 768px) */
        @media (max-width: 767px) {
            .block-container {
                max-width: 100% !important;
                padding-left: 10px !important;
                padding-right: 10px !important;
                padding-top: 0.8rem !important;
            }
            div[data-testid="stHorizontalBlock"] {
                flex-direction: column !important;
                gap: 10px !important;
            }
            div[data-testid="column"] {
                width: 100% !important;
                min-width: 100% !important;
                flex: 1 1 100% !important;
                margin-bottom: 8px !important;
            }
            h1 { font-size: 1.6rem !important; }
            h2 { font-size: 1.35rem !important; }
            h3 { font-size: 1.15rem !important; }
            h4 { font-size: 1.05rem !important; }

            /* Table & Plotly Chart Responsive Wrappers */
            div[data-testid="stTable"], div[data-testid="stDataFrame"], .js-plotly-plot {
                max-width: 100% !important;
                overflow-x: auto !important;
                display: block !important;
            }

            .stButton > button {
                width: 100% !important;
            }
        }
    </style>
""", unsafe_allow_html=True)

# Initialize Session State Variables
if "current_page" not in st.session_state:
    st.session_state.current_page = "landing"
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "user_doc" not in st.session_state:
    st.session_state.user_doc = None
if "auth_mode" not in st.session_state:
    st.session_state.auth_mode = "login"
if "active_tab" not in st.session_state:
    st.session_state.active_tab = "dashboard"
if "reg_success_msg" not in st.session_state:
    st.session_state.reg_success_msg = None

# Handle URL parameters or nav triggers if any
params = st.query_params
if "page" in params:
    target_page = params["page"]
    if target_page in ["landing", "auth"]:
        st.session_state.current_page = target_page
        if "mode" in params and params["mode"] in ["login", "register"]:
            st.session_state.auth_mode = params["mode"]

# =============================================================================
# ROUTER 1: AUTHENTICATED WORKSPACE & MAIN DASHBOARD
# =============================================================================
if st.session_state.authenticated and st.session_state.user_doc:

    active_tab = st.session_state.get("active_tab", "dashboard")

    if active_tab == "dashboard":
        render_main_dashboard()
    elif active_tab == "risk_analysis":
        render_top_navigation()
        render_risk_analysis_page()
    elif active_tab == "visualization":
        render_top_navigation()
        render_visualization_page()
    elif active_tab == "history":
        render_top_navigation()
        render_history_page()
    elif active_tab == "profile":
        render_top_navigation()
        render_profile_page()

# =============================================================================
# ROUTER 2: LANDING PAGE (DEFAULT HOME VIEW)
# =============================================================================
elif st.session_state.current_page == "landing":
    render_landing_page()

# =============================================================================
# ROUTER 3: AUTHENTICATION PORTAL (LOGIN & REGISTER)
# =============================================================================
elif st.session_state.current_page == "auth":
    # Load Custom Auth CSS Styling
    CSS_PATH = os.path.join(os.getcwd(), "css", "auth_style.css")
    if os.path.exists(CSS_PATH):
        with open(CSS_PATH, 'r') as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

    # Header Navigation Back to Landing Page
    top_col1, top_col2 = st.columns([3.5, 1.5])
    with top_col2:
        if st.button("Back to Home", key="btn_back_home", use_container_width=True):
            st.session_state.current_page = "landing"
            st.query_params.clear()
            st.rerun()

    # -------------------------------------------------------------------------
    # LOGIN VIEW
    # -------------------------------------------------------------------------
    if st.session_state.auth_mode == "login":
        st.markdown("""
            <div class="auth-header-wrapper">
                <div class="auth-title">Welcome Back</div>
                <div class="auth-subtitle">Sign in to access your risk forecasting workspace</div>
            </div>
        """, unsafe_allow_html=True)

        # Show notification if redirected after registration
        if st.session_state.reg_success_msg:
            st.success(st.session_state.reg_success_msg)
            st.session_state.reg_success_msg = None

        with st.form("login_form"):
            login_email = st.text_input("Email Address", key="login_email_input", placeholder="name@organization.com")
            login_password = st.text_input("Password", key="login_password_input", type="password", placeholder="••••••••")

            st.markdown("<br>", unsafe_allow_html=True)
            submit_login = st.form_submit_button("Sign In")

        if submit_login:
            if not login_email.strip() or not login_password.strip():
                st.error("Please enter both Email Address and Password.")
            elif not is_valid_email(login_email):
                st.error("Please enter a valid email address.")
            else:
                success, result = authenticate_user(login_email, login_password)

                if success:
                    st.session_state.authenticated = True
                    st.session_state.user_doc = result
                    st.session_state.active_tab = "dashboard"
                    st.success("Signed in successfully!")
                    st.rerun()
                else:
                    st.error(f"Authentication Error: {result}")

        st.markdown("<div style='margin-top: 14px;'></div>", unsafe_allow_html=True)
        if st.button("Create Account", key="goto_reg_btn", use_container_width=True):
            st.session_state.auth_mode = "register"
            st.session_state.current_page = "auth"
            st.query_params["page"] = "auth"
            st.query_params["mode"] = "register"
            st.rerun()

    # -------------------------------------------------------------------------
    # REGISTER VIEW
    # -------------------------------------------------------------------------
    elif st.session_state.auth_mode == "register":
        st.markdown("""
            <div class="auth-header-wrapper">
                <div class="auth-title">Create Account</div>
                <div class="auth-subtitle">Fill in your information to set up your profile</div>
            </div>
        """, unsafe_allow_html=True)

        with st.form("register_form"):
            # 1. Personal Identity Fields
            st.markdown("##### 1. Personal Information")
            first_name = st.text_input("First Name *", key="reg_fn", placeholder="Alex")
            last_name = st.text_input("Last Name *", key="reg_ln", placeholder="Morgan")
            reg_email = st.text_input("Email Address *", key="reg_email", placeholder="alex.morgan@company.com")

            # 2. Organization & Education Background
            st.markdown("##### 2. Background & Classification")
            org_type = st.selectbox(
                "Organization Type *",
                key="reg_org",
                options=[
                    "Startup",
                    "Multinational Corporation (MNC)",
                    "Small & Medium Enterprise (SME)",
                    "R&D Research Lab",
                    "Educational Institution",
                    "Non-Profit Organization",
                    "Government Agency",
                    "Freelance / Independent",
                    "Tech Incubator / Accelerator"
                ]
            )

            edu_category = st.selectbox(
                "Education / Occupation Category *",
                key="reg_edu",
                options=[
                    "College / University Student",
                    "School Student",
                    "Working Professional / Corporate",
                    "Research Scholar"
                ]
            )

            # Dynamic Fields
            school_name = ""
            standard = ""
            university_name = ""
            degree = ""
            academic_year = ""
            designation = ""
            experience_level = ""

            if "School Student" in edu_category:
                st.markdown("###### School Details")
                standard = st.selectbox("Standard / Grade *", ["8th Grade", "9th Grade", "10th Grade", "11th Grade", "12th Grade"], key="reg_std")
                school_name = st.text_input("School Name *", placeholder="St. Xavier's High School", key="reg_sch")

            elif "College" in edu_category or "Research Scholar" in edu_category:
                st.markdown("###### Higher Education Details")
                degree = st.selectbox("Degree / Major *", ["B.Tech / B.E", "B.Sc", "BCA", "M.Tech", "MBA", "Ph.D", "M.Sc", "MCA", "Diploma", "Other"], key="reg_deg")
                academic_year = st.selectbox("Academic Year *", ["1st Year", "2nd Year", "3rd Year", "4th Year", "Post Graduate", "Alumni"], key="reg_yr")
                university_name = st.text_input("Institution Name *", placeholder="Stanford University", key="reg_univ")

            elif "Working Professional" in edu_category:
                st.markdown("###### Corporate Career Details")
                designation = st.selectbox(
                    "Job Role / Designation *",
                    [
                        "AI/ML Engineer",
                        "Senior Project Manager",
                        "Data Analyst / Scientist",
                        "Software Developer",
                        "System Architect",
                        "Executive / Director",
                        "Quality Assurance Lead",
                        "Business Analyst",
                        "Management Consultant",
                        "Other Professional"
                    ],
                    key="reg_desig"
                )
                experience_level = st.selectbox("Experience Level *", ["Entry Level (0-2 yrs)", "Mid-Level (3-5 yrs)", "Senior Level (6-10 yrs)", "Lead / Executive (10+ yrs)"], key="reg_exp")

            # 3. Security Passwords
            st.markdown("##### 3. Account Passwords")
            reg_password = st.text_input("Password *", key="reg_pwd", type="password", placeholder="••••••••")
            confirm_password = st.text_input("Confirm Password *", key="reg_cpwd", type="password", placeholder="••••••••")

            st.markdown("<br>", unsafe_allow_html=True)
            submit_register = st.form_submit_button("Complete Registration")

        if submit_register:
            valid, val_msg = validate_registration_fields(first_name, last_name, reg_email, reg_password, confirm_password)

            if not valid:
                st.error(val_msg)
            else:
                user_payload = {
                    "first_name": first_name,
                    "last_name": last_name,
                    "email": reg_email,
                    "password": reg_password,
                    "organization_type": org_type,
                    "education_category": edu_category,
                    "school_name": school_name,
                    "standard": standard,
                    "university_name": university_name,
                    "degree": degree,
                    "academic_year": academic_year,
                    "designation": designation,
                    "experience_level": experience_level
                }

                reg_success, reg_result = register_user(user_payload)

                if reg_success:
                    st.balloons()
                    # AUTOMATIC REDIRECT TO SIGN IN PAGE ON REGISTRATION SUCCESS
                    st.session_state.auth_mode = "login"
                    st.session_state.reg_success_msg = f"Account registered successfully for {reg_email}! Please sign in using your credentials."
                    st.query_params["page"] = "auth"
                    st.query_params["mode"] = "login"
                    st.rerun()
                else:
                    st.error(f"Registration Error: {reg_result}")

        st.markdown("<div style='margin-top: 14px;'></div>", unsafe_allow_html=True)
        if st.button("Sign In", key="goto_login_btn", use_container_width=True):
            st.session_state.auth_mode = "login"
            st.session_state.current_page = "auth"
            st.query_params["page"] = "auth"
            st.query_params["mode"] = "login"
            st.rerun()
