"""
Landing Page Component for AI-Based Project Risk Forecasting System
Renders the complete enterprise landing page with 3 distinct card styles & custom animation transitions for Features, How It Works, and Tech Stack.
"""

import os
import streamlit as st


def render_landing_page():
    """Renders the landing page with single-line Navbar, Hero, About, Features (Type 1 Cards), How It Works (Type 2 Cards), Tech Stack (Type 3 Cards), About Project, and Footer."""

    # Load Landing Page Custom CSS
    css_path = os.path.join(os.getcwd(), "css", "landing_style.css")
    if os.path.exists(css_path):
        with open(css_path, "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # 1. NAVBAR (SINGLE LINE, UNIFORM BUTTONS)
    # -------------------------------------------------------------------------
    st.markdown("""<div class="landing-navbar-container"><div class="nav-brand">🛡️ AI Risk Forecasting</div><div class="nav-links-row"><a href="#hero" class="nav-btn" target="_self">Home</a><a href="#features" class="nav-btn" target="_self">Features</a><a href="#how-it-works" class="nav-btn" target="_self">How It Works</a><a href="#about" class="nav-btn" target="_self">About</a><a href="?page=auth&mode=login" class="nav-btn" target="_self">Login</a><a href="?page=auth&mode=register" class="nav-btn" target="_self">Get Started</a></div></div>""", unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # 2. HERO CARD (WITH FLOATING PILL & BORDER GLOW ANIMATION)
    # -------------------------------------------------------------------------
    st.markdown("""<div class="advanced-hero-card" id="hero"><span class="hero-pill-animated">✨ Intelligent Predictive Risk Analytics</span><h1 class="hero-title-gradient">AI-Based Project Risk Forecasting System</h1><div class="hero-subtitle-bold">Predict Project Risks Before They Become Problems</div><p class="hero-desc-lead">An intelligent machine learning platform that analyzes project data, forecasts potential risks, and helps teams make proactive decisions for successful project delivery.</p></div>""", unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # 3. ABOUT SECTION (GLASSMORPHISM SPOTLIGHT CARD WITH FACTOR PILLS)
    # -------------------------------------------------------------------------
    st.markdown("""<div class="section-wrapper" id="about"><div class="section-header"><div class="section-tag">About System</div><h2 class="section-title">Smarter Project Risk Management</h2></div><div class="about-glass-card"><p class="about-paragraph">Project risks such as delays, resource constraints, budget issues, and changing requirements can affect successful project delivery.</p><p class="about-paragraph">Our AI-Based Project Risk Forecasting System analyzes project-related data using machine learning to identify patterns and forecast potential risk levels, helping teams understand project challenges and take preventive action.</p><div class="risk-factors-pills"><span class="factor-pill">⏱️ Project Delays</span><span class="factor-pill">👥 Resource Constraints</span><span class="factor-pill">💰 Budget Overruns</span><span class="factor-pill">🔄 Requirement Changes</span><span class="factor-pill">🤝 Vendor Dependencies</span></div></div></div>""", unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # 4. SECTION 1: FEATURES (TYPE 1 CARDS - FLOATING BENTO GRID WITH 3D ROTATE BADGES)
    # -------------------------------------------------------------------------
    st.markdown("""<div class="section-wrapper" id="features"><div class="section-header"><div class="section-tag">Key Features</div><h2 class="section-title">Intelligent Risk Forecasting</h2><p class="section-desc">Use machine learning to analyze project information and forecast potential project risks.</p></div><div class="features-grid-3col"><div class="type1-feature-card"><div class="type1-card-accent"></div><div class="type1-icon-badge badge-violet">🔍</div><div class="type1-title">Early Risk Identification</div><div class="type1-desc">Identify potential risks before they become critical and affect project progress.</div></div><div class="type1-feature-card"><div class="type1-card-accent"></div><div class="type1-icon-badge badge-emerald">📊</div><div class="type1-title">Risk Analysis</div><div class="type1-desc">Understand the project factors that contribute to different levels of risk.</div></div><div class="type1-feature-card"><div class="type1-card-accent"></div><div class="type1-icon-badge badge-amber">💡</div><div class="type1-title">Data-Driven Insights</div><div class="type1-desc">Turn project data into meaningful insights that support better decision-making.</div></div><div class="type1-feature-card"><div class="type1-card-accent"></div><div class="type1-icon-badge badge-cyan">📈</div><div class="type1-title">Project Monitoring</div><div class="type1-desc">Keep track of project information and risk prediction results in one place.</div></div><div class="type1-feature-card"><div class="type1-card-accent"></div><div class="type1-icon-badge badge-rose">📜</div><div class="type1-title">Prediction History</div><div class="type1-desc">Review previous risk predictions to understand project risk patterns over time.</div></div><div class="type1-feature-card"><div class="type1-card-accent"></div><div class="type1-icon-badge badge-blue">⚡</div><div class="type1-title">Intelligent Risk Forecasting</div><div class="type1-desc">Use machine learning to analyze project information and forecast potential project risks.</div></div></div></div>""", unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # 5. SECTION 2: HOW IT WORKS (TYPE 2 CARDS - HORIZONTAL SLIDE TIMELINE PROCESS CARDS)
    # -------------------------------------------------------------------------
    st.markdown("""<div class="section-wrapper" id="how-it-works"><div class="section-header"><div class="section-tag">Workflow</div><h2 class="section-title">How It Works</h2></div><div class="steps-grid-4col"><div class="type2-step-card"><span class="type2-step-badge">STEP 01</span><div class="type2-title">Project Data</div><div class="type2-desc">Provide the required project information for analysis.</div></div><div class="type2-step-card"><span class="type2-step-badge">STEP 02</span><div class="type2-title">Data Processing</div><div class="type2-desc">The system processes and analyzes the project data to identify important risk-related patterns.</div></div><div class="type2-step-card"><span class="type2-step-badge">STEP 03</span><div class="type2-title">Risk Forecasting</div><div class="type2-desc">The trained machine learning model evaluates the project and forecasts its potential risk level.</div></div><div class="type2-step-card"><span class="type2-step-badge">STEP 04</span><div class="type2-title">Risk Insights</div><div class="type2-desc">The system presents the predicted risk level and relevant analysis to help users make informed decisions.</div></div></div></div>""", unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # 6. SECTION 3: TECHNOLOGY STACK (TYPE 3 CARDS - GLASSMORPHISM GLOW TECH CARDS WITH SCALE ZOOM)
    # -------------------------------------------------------------------------
    st.markdown("""<div class="section-wrapper" id="technology"><div class="section-header"><div class="section-tag">Tech Stack</div><h2 class="section-title">Built with Modern Technology</h2></div><div class="tech-grid-4col"><div class="type3-tech-card"><span class="type3-tech-pill">Machine Learning</span><div class="type3-desc">Analyze project data and identify risk patterns.</div></div><div class="type3-tech-card"><span class="type3-tech-pill">CatBoost AI</span><div class="type3-desc">Our trained CatBoost model is used for project risk forecasting.</div></div><div class="type3-tech-card"><span class="type3-tech-pill">Python Engine</span><div class="type3-desc">Used for data processing, machine learning, and application development.</div></div><div class="type3-tech-card"><span class="type3-tech-pill">SQLite Database</span><div class="type3-desc">Used for reliable, zero-latency real-time data management.</div></div></div></div>""", unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # 7. ABOUT THE PROJECT SECTION (DARK SPOTLIGHT CARD)
    # -------------------------------------------------------------------------
    st.markdown("""<div class="section-wrapper" id="about-project"><div class="proactive-card"><h2 class="proactive-title">Building a More Proactive Approach to Project Risk</h2><p class="proactive-text">Traditional project management often identifies risks after they have already started affecting project progress.</p><p class="proactive-text">Our system focuses on early risk forecasting, allowing teams to understand potential problems sooner and make proactive decisions.</p><p class="proactive-text" style="font-weight: 600; color: #ffffff; margin-top: 18px;">The goal is simple:</p><div class="flow-box"><div class="flow-pipeline">Identify risks early &nbsp;→&nbsp; Understand their impact &nbsp;→&nbsp; Take action &nbsp;→&nbsp; Improve project outcomes</div></div></div></div>""", unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # 8. FOOTER SECTION
    # -------------------------------------------------------------------------
    st.markdown("""<div class="landing-footer"><div class="footer-brand">AI-Based Project Risk Forecasting System</div><div class="footer-tagline">Predict risks. Plan better. Deliver successfully.</div><div class="footer-copyright">© 2026 AI-Based Project Risk Forecasting System</div></div>""", unsafe_allow_html=True)
