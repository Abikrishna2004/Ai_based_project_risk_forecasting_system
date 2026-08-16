"""
User Profile Component for AI-Based Project Risk Forecasting System
Implements Profile Information, Account Overview metrics, Profile Update, and Password Security settings.
"""

import streamlit as st
from utils.api_client import (
    get_user_dashboard_metrics,
    get_user_profile,
    update_user_profile,
    change_password,
    upload_profile_image
)


def render_profile_page():
    """Renders user profile interface according to exact user specification."""
    user = st.session_state.get("user_doc", {})
    user_id = user.get("_id", "")

    # Fetch latest profile from API if available
    latest_user = get_user_profile(user_id)
    if latest_user:
        user = latest_user
        st.session_state["user_doc"] = latest_user

    full_name = f"{user.get('first_name', '')} {user.get('last_name', '')}".strip() or "User"
    email = user.get("email", "N/A")
    created_at = user.get("created_at", "N/A")
    org_type = user.get("organization_type", "Startup")
    edu_cat = user.get("education_category", "College / University Student")
    designation = user.get("designation", "N/A")
    avatar_url = user.get("profile_image")

    # Fetch real-time account metrics
    metrics = get_user_dashboard_metrics(user_id)
    total_projects = metrics["total_projects"]
    high_risk = metrics["high_risk_count"]
    medium_risk = metrics["medium_risk_count"]
    low_risk = metrics["low_risk_count"]

    # -------------------------------------------------------------------------
    # HEADER & SUBTITLE
    # -------------------------------------------------------------------------
    st.markdown("""
        <div style="margin-bottom: 24px;">
            <h2 style="font-size: 1.8rem; font-weight: 800; color: var(--text-primary);">USER PROFILE</h2>
            <p style="color: var(--text-secondary); font-size: 1rem;">
                Manage your account credentials, security settings, and organization details.
            </p>
        </div>
    """, unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # 1. PROFILE INFORMATION CARD
    # -------------------------------------------------------------------------
    st.markdown("### PROFILE INFORMATION")
    p_col1, p_col2 = st.columns([1, 3])

    with p_col1:
        st.markdown(f"""
            <div style="background: var(--bg-card); border: 1px solid var(--border-color); border-radius: 16px; padding: 20px; text-align: center; box-shadow: var(--card-shadow);">
                <div style="font-size: 3.5rem; margin-bottom: 8px;">👤</div>
                <div style="font-size: 1.15rem; font-weight: 800; color: var(--text-primary);">{full_name}</div>
                <div style="font-size: 0.85rem; color: var(--text-secondary);">{designation or 'Member'}</div>
            </div>
        """, unsafe_allow_html=True)

    with p_col2:
        st.markdown(f"""
            <div style="background: var(--bg-card); border: 1px solid var(--border-color); border-radius: 16px; padding: 24px; box-shadow: var(--card-shadow);">
                <div style="margin-bottom: 12px;">
                    <div style="font-size: 0.85rem; font-weight: 600; color: var(--text-secondary);">Full Name</div>
                    <div style="font-size: 1.2rem; font-weight: 800; color: var(--text-primary);">{full_name}</div>
                </div>
                <div style="margin-bottom: 12px;">
                    <div style="font-size: 0.85rem; font-weight: 600; color: var(--text-secondary);">Email Address</div>
                    <div style="font-size: 1.05rem; font-weight: 700; color: var(--text-primary);">{email}</div>
                </div>
                <div style="margin-bottom: 12px;">
                    <div style="font-size: 0.85rem; font-weight: 600; color: var(--text-secondary);">Organization & Category</div>
                    <div style="font-size: 1rem; font-weight: 600; color: var(--text-primary);">{org_type} • {edu_cat}</div>
                </div>
                <div>
                    <div style="font-size: 0.85rem; font-weight: 600; color: var(--text-secondary);">Member Since</div>
                    <div style="font-size: 0.95rem; font-weight: 600; color: var(--text-primary);">{created_at}</div>
                </div>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # 2. ACCOUNT OVERVIEW METRICS
    # -------------------------------------------------------------------------
    st.markdown("### ACCOUNT OVERVIEW")
    st.markdown("<br>", unsafe_allow_html=True)

    m_col1, m_col2, m_col3, m_col4 = st.columns(4)
    with m_col1:
        st.metric("Projects Analyzed", total_projects)
    with m_col2:
        st.metric("High Risk Projects", high_risk)
    with m_col3:
        st.metric("Medium Risk Projects", medium_risk)
    with m_col4:
        st.metric("Low Risk Projects", low_risk)

    st.markdown("<br><br>", unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # 3. ACCOUNT SETTINGS & UPDATE FORMS
    # -------------------------------------------------------------------------
    st.markdown("### ACCOUNT SETTINGS")
    tab_edit, tab_pwd, tab_img = st.tabs(["✏️ Edit Profile", "🔑 Change Password", "🖼️ Profile Image"])

    with tab_edit:
        with st.form("form_edit_profile"):
            st.markdown("##### Update Profile Details")
            ef_name = st.text_input("First Name", value=user.get("first_name", ""))
            el_name = st.text_input("Last Name", value=user.get("last_name", ""))
            e_org = st.selectbox(
                "Organization Type",
                ["Startup", "Enterprise", "Educational Institution", "Government", "Non-Profit", "Freelancer / Individual"],
                index=["Startup", "Enterprise", "Educational Institution", "Government", "Non-Profit", "Freelancer / Individual"].index(org_type) if org_type in ["Startup", "Enterprise", "Educational Institution", "Government", "Non-Profit", "Freelancer / Individual"] else 0
            )
            e_desig = st.text_input("Designation / Role", value=user.get("designation", ""))

            btn_update_prof = st.form_submit_button("Save Profile Changes", use_container_width=True)

            if btn_update_prof:
                update_payload = {
                    "first_name": ef_name.strip(),
                    "last_name": el_name.strip(),
                    "organization_type": e_org,
                    "designation": e_desig.strip()
                }
                ok, msg, updated_u = update_user_profile(user_id, update_payload)
                if ok:
                    st.success(msg)
                    if updated_u:
                        st.session_state["user_doc"] = updated_u
                    st.rerun()
                else:
                    st.error(msg)

    with tab_pwd:
        with st.form("form_change_password"):
            st.markdown("##### Change Account Password")
            old_p = st.text_input("Current Password", type="password")
            new_p = st.text_input("New Password", type="password")
            conf_p = st.text_input("Confirm New Password", type="password")

            btn_change_p = st.form_submit_button("Update Password", use_container_width=True)

            if btn_change_p:
                if not old_p or not new_p:
                    st.error("Please enter both current and new passwords.")
                elif new_p != conf_p:
                    st.error("New password and confirm password do not match.")
                elif len(new_p) < 6:
                    st.error("New password must be at least 6 characters long.")
                else:
                    ok, msg = change_password(user_id, old_p, new_p)
                    if ok:
                        st.success(msg)
                    else:
                        st.error(msg)

    with tab_img:
        st.markdown("##### Upload Profile Avatar Image")
        img_file = st.file_uploader("Choose an image (PNG, JPG, JPEG)", type=["png", "jpg", "jpeg"])
        if img_file is not None:
            if st.button("Upload Image", key="btn_upload_img", use_container_width=True):
                img_bytes = img_file.read()
                ok, msg, url = upload_profile_image(user_id, img_bytes, img_file.name)
                if ok:
                    st.success(msg)
                    st.rerun()
                else:
                    st.error(msg)
