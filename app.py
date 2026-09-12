import streamlit as st
import json
import os
import re
import numpy as np

import faiss
import cv2
from sentence_transformers import SentenceTransformer
from datetime import datetime, timezone

from dotenv import load_dotenv
from google import genai
from google.genai import types
from auth import register_user, login_user
from database import chat_history_collection, technician_records_collection


# ============================================================
# LOAD ENVIRONMENT / GEMINI
# ============================================================

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    st.error("Gemini API key not found. Please check your .env file.")
    st.stop()

client = genai.Client(api_key=api_key)


# ============================================================
# GEMINI MODELS
# ============================================================

models_to_try = [
    "gemini-3.8-flash",
    "gemini-3.7-flash",
    "gemini-3.6-flash"
]


# ============================================================
# PAGE CONFIGURATION & DESIGN SYSTEM
# ============================================================

st.set_page_config(
    page_title="LabAI — The Intelligent AI Copilot for Engineering Labs",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Design System (CSS)
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

/* Global Reset & Base */
html, body, [class*="css"], .stApp {
    font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif !important;
    color: #F1F5F9;
}

.stApp {
    background: radial-gradient(circle at 50% 0%, #171d36 0%, #0c101c 45%, #07090f 100%) !important;
    min-height: 100vh;
}

/* Hide standard Streamlit header & decoration */
header[data-testid="stHeader"] {
    visibility: hidden !important;
    display: none !important;
}

[data-testid="stToolbar"] {
    visibility: hidden !important;
    display: none !important;
}

#MainMenu, footer {
    visibility: hidden !important;
    display: none !important;
}

.block-container {
    padding-top: 1.2rem !important;
    padding-bottom: 3.5rem !important;
    max-width: 1240px !important;
}

/* Typography & Colors */
.gradient-text {
    background: linear-gradient(135deg, #A5B4FC 0%, #818CF8 35%, #38BDF8 70%, #C084FC 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    display: inline-block;
}

.gradient-text-gold {
    background: linear-gradient(135deg, #FDE68A 0%, #F59E0B 50%, #EF4444 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    display: inline-block;
}

/* Navbar Component */
.nav-container {
    display: flex;
    align-items: center;
    justify-content: space-between;
    background: rgba(15, 23, 42, 0.85);
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 16px;
    padding: 14px 24px;
    margin-bottom: 2rem;
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.35);
}

.nav-brand {
    display: flex;
    align-items: center;
    gap: 12px;
}

.brand-title {
    font-size: 1.5rem;
    font-weight: 800;
    letter-spacing: -0.5px;
    margin: 0;
    color: #FFFFFF;
    display: flex;
    align-items: center;
    gap: 10px;
}

.status-pill {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(16, 185, 129, 0.12);
    border: 1px solid rgba(16, 185, 129, 0.3);
    color: #34D399;
    font-size: 0.72rem;
    font-weight: 600;
    padding: 4px 10px;
    border-radius: 9999px;
    letter-spacing: 0.5px;
    text-transform: uppercase;
}

.pulse-dot {
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background-color: #10B981;
    box-shadow: 0 0 10px #10B981;
    animation: pulseGlow 2s infinite;
}

@keyframes pulseGlow {
    0% { transform: scale(0.95); opacity: 0.8; }
    50% { transform: scale(1.3); opacity: 1; box-shadow: 0 0 12px #34D399; }
    100% { transform: scale(0.95); opacity: 0.8; }
}

/* Hero Section */
.hero-wrapper {
    text-align: center;
    padding: 30px 10px 20px 10px;
    position: relative;
}

.hero-pill {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    background: rgba(99, 102, 241, 0.12);
    border: 1px solid rgba(129, 140, 248, 0.3);
    color: #C7D2FE;
    font-size: 0.82rem;
    font-weight: 700;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    padding: 8px 22px;
    border-radius: 9999px;
    margin-bottom: 24px;
    box-shadow: 0 0 24px rgba(99, 102, 241, 0.2);
}

.hero-title {
    font-size: 3.8rem;
    font-weight: 800;
    line-height: 1.12;
    letter-spacing: -1.2px;
    color: #FFFFFF;
    margin: 0 auto 20px auto;
    max-width: 980px;
}

@media (max-width: 768px) {
    .hero-title {
        font-size: 2.3rem;
    }
}

.hero-subtitle {
    font-size: 1.25rem;
    line-height: 1.6;
    color: #94A3B8;
    max-width: 820px;
    margin: 0 auto 36px auto;
    font-weight: 400;
}

/* Live Workbench Terminal Showcase */
.showcase-window {
    background: #0B101D;
    border: 1px solid rgba(255, 255, 255, 0.12);
    border-radius: 18px;
    overflow: hidden;
    margin: 36px 0;
    box-shadow: 0 25px 60px -15px rgba(0, 0, 0, 0.8), 0 0 40px rgba(99, 102, 241, 0.12);
}

.window-header {
    background: #131A2E;
    padding: 12px 20px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    border-bottom: 1px solid rgba(255, 255, 255, 0.08);
}

.window-dots {
    display: flex;
    gap: 8px;
}

.dot {
    width: 11px;
    height: 11px;
    border-radius: 50%;
    display: inline-block;
}
.dot-red { background: #EF4444; }
.dot-yellow { background: #F59E0B; }
.dot-green { background: #10B981; }

.window-title {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.8rem;
    color: #94A3B8;
}

.showcase-body {
    padding: 24px;
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 20px;
    text-align: left;
}

@media (max-width: 768px) {
    .showcase-body {
        grid-template-columns: 1fr;
    }
}

.demo-pane {
    background: rgba(18, 25, 43, 0.7);
    border: 1px solid rgba(255, 255, 255, 0.06);
    border-radius: 14px;
    padding: 20px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.82rem;
    line-height: 1.6;
}

.pane-badge {
    display: inline-block;
    padding: 4px 10px;
    border-radius: 6px;
    font-size: 0.72rem;
    font-weight: 700;
    margin-bottom: 14px;
    letter-spacing: 0.5px;
}

.pane-badge-vision {
    background: rgba(56, 189, 248, 0.15);
    color: #38BDF8;
    border: 1px solid rgba(56, 189, 248, 0.3);
}

.pane-badge-rag {
    background: rgba(168, 85, 247, 0.15);
    color: #C084FC;
    border: 1px solid rgba(168, 85, 247, 0.3);
}

.code-green { color: #34D399; font-weight: 600; }
.code-amber { color: #FBBF24; font-weight: 600; }
.code-red { color: #F87171; font-weight: 600; }
.code-cyan { color: #38BDF8; font-weight: 600; }
.code-muted { color: #64748B; }

/* Metric Cards */
.metric-card {
    background: rgba(20, 27, 45, 0.7);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 16px;
    padding: 24px 16px;
    text-align: center;
    transition: all 0.3s ease;
    backdrop-filter: blur(10px);
}

.metric-card:hover {
    border-color: rgba(99, 102, 241, 0.4);
    transform: translateY(-3px);
    background: rgba(30, 41, 69, 0.75);
    box-shadow: 0 12px 28px rgba(0, 0, 0, 0.4);
}

.metric-val {
    font-size: 2.6rem;
    font-weight: 800;
    letter-spacing: -1px;
    margin-bottom: 4px;
    color: #FFFFFF;
}

.metric-lbl {
    font-size: 0.85rem;
    font-weight: 600;
    color: #94A3B8;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

/* Feature Grid */
.feature-card {
    background: rgba(17, 24, 39, 0.65);
    backdrop-filter: blur(14px);
    -webkit-backdrop-filter: blur(14px);
    border: 1px solid rgba(255, 255, 255, 0.07);
    border-radius: 18px;
    padding: 26px;
    height: 100%;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    position: relative;
    overflow: hidden;
    text-align: left;
}

.feature-card::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 3px;
    background: linear-gradient(90deg, transparent, rgba(99, 102, 241, 0.7), transparent);
    opacity: 0;
    transition: opacity 0.3s ease;
}

.feature-card:hover {
    transform: translateY(-4px);
    border-color: rgba(99, 102, 241, 0.35);
    box-shadow: 0 20px 40px -10px rgba(0, 0, 0, 0.6);
}

.feature-card:hover::before {
    opacity: 1;
}

.feature-icon-wrapper {
    width: 48px;
    height: 48px;
    border-radius: 12px;
    background: rgba(99, 102, 241, 0.15);
    border: 1px solid rgba(129, 140, 248, 0.25);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.5rem;
    margin-bottom: 18px;
}

.feature-title {
    font-size: 1.2rem;
    font-weight: 700;
    color: #FFFFFF;
    margin-bottom: 10px;
}

.feature-desc {
    font-size: 0.92rem;
    color: #94A3B8;
    line-height: 1.6;
    margin-bottom: 16px;
}

.feature-tag {
    display: inline-block;
    font-size: 0.72rem;
    font-weight: 600;
    color: #818CF8;
    background: rgba(99, 102, 241, 0.12);
    border: 1px solid rgba(99, 102, 241, 0.25);
    border-radius: 6px;
    padding: 3px 8px;
}

/* Workflow Steps */
.step-card {
    background: rgba(18, 25, 42, 0.7);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 16px;
    padding: 26px;
    position: relative;
    text-align: left;
    height: 100%;
}

.step-num {
    font-family: 'JetBrains Mono', monospace;
    font-size: 2.2rem;
    font-weight: 800;
    color: rgba(99, 102, 241, 0.4);
    line-height: 1;
    margin-bottom: 14px;
}

/* Comparison Table */
.comp-card {
    border-radius: 16px;
    padding: 26px;
    height: 100%;
    text-align: left;
}

.comp-bad {
    background: rgba(239, 68, 68, 0.05);
    border: 1px solid rgba(239, 68, 68, 0.2);
}

.comp-good {
    background: rgba(16, 185, 129, 0.06);
    border: 1px solid rgba(16, 185, 129, 0.3);
    box-shadow: 0 0 30px rgba(16, 185, 129, 0.08);
}

.comp-title {
    font-size: 1.25rem;
    font-weight: 700;
    margin-bottom: 18px;
    display: flex;
    align-items: center;
    gap: 8px;
}

.comp-item {
    font-size: 0.92rem;
    line-height: 1.7;
    margin-bottom: 12px;
    color: #CBD5E1;
}

/* Auth Card Styling */
.auth-box {
    background: rgba(17, 24, 39, 0.95);
    backdrop-filter: blur(20px);
    border: 1px solid rgba(99, 102, 241, 0.3);
    border-radius: 22px;
    padding: 34px;
    box-shadow: 0 25px 60px -15px rgba(0, 0, 0, 0.8), 0 0 40px rgba(99, 102, 241, 0.2);
}

/* Streamlit Widget Styling Overrides */
.stButton > button {
    background: linear-gradient(135deg, #4F46E5 0%, #6366F1 50%, #4338CA 100%) !important;
    color: #FFFFFF !important;
    border: 1px solid rgba(255, 255, 255, 0.15) !important;
    border-radius: 12px !important;
    font-weight: 600 !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-size: 0.95rem !important;
    box-shadow: 0 4px 20px rgba(99, 102, 241, 0.35) !important;
    transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1) !important;
}

.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 30px rgba(99, 102, 241, 0.6) !important;
    border-color: rgba(255, 255, 255, 0.35) !important;
}

.stTextInput input, .stSelectbox div[data-baseweb="select"] {
    background-color: rgba(15, 23, 42, 0.85) !important;
    border: 1px solid rgba(255, 255, 255, 0.12) !important;
    border-radius: 10px !important;
    color: #F8FAFC !important;
}

.stTextInput input:focus {
    border-color: #818CF8 !important;
    box-shadow: 0 0 0 2px rgba(129, 140, 248, 0.25) !important;
}

[data-testid="stChatInput"] textarea {
    color: #000000 !important;
    caret-color: #000000 !important;
}

/* Footer */
.footer-wrap {
    border-top: 1px solid rgba(255, 255, 255, 0.08);
    padding: 40px 0 20px 0;
    margin-top: 60px;
    text-align: center;
    color: #64748B;
    font-size: 0.88rem;
}

.footer-badges {
    display: flex;
    justify-content: center;
    gap: 10px;
    flex-wrap: wrap;
    margin-bottom: 20px;
}

.tech-badge {
    background: rgba(30, 41, 59, 0.6);
    border: 1px solid rgba(255, 255, 255, 0.08);
    color: #94A3B8;
    padding: 5px 12px;
    border-radius: 8px;
    font-size: 0.78rem;
    font-weight: 500;
}
</style>
""", unsafe_allow_html=True)


# ============================================================
# SESSION STATE INITIALIZATION
# ============================================================

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "user_name" not in st.session_state:
    st.session_state.user_name = None

if "user_role" not in st.session_state:
    st.session_state.user_role = None

if "user_id" not in st.session_state:
    st.session_state.user_id = None

if "show_auth" not in st.session_state:
    st.session_state.show_auth = False

if "auth_tab" not in st.session_state:
    st.session_state.auth_tab = "Login"


# ============================================================
# AUTHENTICATION FORM COMPONENT
# ============================================================

def render_auth_component(key_prefix="auth"):
    st.markdown("""
    <div style="text-align: center; margin-bottom: 24px;">
        <h2 style="margin: 0; color: #FFFFFF; font-weight: 800; font-size: 1.8rem;">
            Access <span class="gradient-text">LabAI</span> Workspace
        </h2>
        <p style="color: #94A3B8; margin-top: 6px; font-size: 0.95rem;">
            Sign in or explore our public lab repository immediately
        </p>
    </div>
    """, unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["🔑 Sign In", "✨ Create Account", "⚡ Instant Guest Access"])

    with tab1:
        st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
        login_email = st.text_input("📧 Institutional or Personal Email", key=f"{key_prefix}_login_email")
        login_password = st.text_input("🔒 Password", type="password", key=f"{key_prefix}_login_pass")

        if st.button("🚀 Sign In to LabAI", use_container_width=True, key=f"{key_prefix}_login_btn"):
            if not login_email or not login_password:
                st.warning("Please provide both email and password.")
            else:
                with st.spinner("Authenticating..."):
                    success, user = login_user(login_email, login_password)
                if success:
                    st.session_state.logged_in = True
                    st.session_state.user_name = user.get("name", "Student")
                    st.session_state.user_role = user.get("role", "student")
                    st.session_state.user_id = str(user.get("_id", ""))
                    st.session_state.show_auth = False
                    st.success(f"Welcome back, {st.session_state.user_name}!")
                    st.rerun()
                else:
                    st.error("Invalid credentials. Check your email or password, or try Instant Guest Access.")

    with tab2:
        st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
        reg_role = st.selectbox(
            "🎯 Select Your Role",
            ["Student", "Lab Technician", "Public Researcher"],
            key=f"{key_prefix}_reg_role"
        )
        reg_name = st.text_input("👤 Full Name", key=f"{key_prefix}_reg_name")
        reg_email = st.text_input("📧 Email Address", key=f"{key_prefix}_reg_email")
        reg_password = st.text_input("🔒 Choose Password", type="password", key=f"{key_prefix}_reg_pass")

        if st.button("✨ Create Account & Enter", use_container_width=True, key=f"{key_prefix}_reg_btn"):
            if not reg_name or not reg_email or not reg_password:
                st.warning("Please complete all registration fields.")
            else:
                with st.spinner("Registering user account..."):
                    role_str = "technician" if reg_role == "Lab Technician" else ("public" if reg_role == "Public Researcher" else "student")
                    success, message = register_user(reg_name, reg_email, reg_password, role_str)
                if success:
                    st.success("Account created successfully! You may now sign in or access as guest.")
                else:
                    st.error(message)

    with tab3:
        st.markdown("""
        <div style="background: rgba(99, 102, 241, 0.1); border: 1px solid rgba(99, 102, 241, 0.25); border-radius: 12px; padding: 18px; margin: 15px 0;">
            <p style="color: #E2E8F0; font-size: 0.95rem; margin-bottom: 6px; font-weight: 600;">
                🎓 One-Click Public Repository Access
            </p>
            <p style="color: #94A3B8; font-size: 0.86rem; line-height: 1.5; margin: 0;">
                Instantly explore all 7 Communication Systems experiments, interactive RAG manual queries, and electronics circuit diagnostic workflows without registration.
            </p>
        </div>
        """, unsafe_allow_html=True)

        if st.button("⚡ Enter as Guest Explorer", use_container_width=True, key=f"{key_prefix}_guest_btn"):
            st.session_state.logged_in = True
            st.session_state.user_role = "public"
            st.session_state.user_name = "Guest Explorer"
            st.session_state.user_id = "guest_temp"
            st.session_state.show_auth = False
            st.rerun()


# ============================================================
# PROFESSIONAL LANDING PAGE
# ============================================================

def legacy_show_landing_page():
    # --------------------------------------------------------
    # 1. TOP NAVBAR
    # --------------------------------------------------------
    c_brand, c_guest, c_auth, c_action = st.columns([5, 2.2, 2.2, 2.4])

    with c_brand:
        st.markdown("""
        <div class="brand-title">
            <span>🔬 LabAI</span>
            <span class="status-pill"><span class="pulse-dot"></span> GEMINI 3 & FAISS ONLINE</span>
        </div>
        """, unsafe_allow_html=True)

    with c_guest:
        if st.button("⚡ Instant Demo", key="nav_btn_guest", use_container_width=True):
            st.session_state.logged_in = True
            st.session_state.user_role = "public"
            st.session_state.user_name = "Guest Explorer"
            st.session_state.user_id = "guest_demo"
            st.session_state.show_auth = False
            st.rerun()

    with c_auth:
        if not st.session_state.show_auth:
            if st.button("🔑 Sign In", key="nav_btn_signin", use_container_width=True):
                st.session_state.show_auth = True
                st.session_state.auth_tab = "Login"
                st.rerun()
        else:
            if st.button("✖ Close Login", key="nav_btn_close", use_container_width=True):
                st.session_state.show_auth = False
                st.rerun()

    with c_action:
        if st.button("🚀 Launch Lab", key="nav_btn_launch", use_container_width=True):
            st.session_state.show_auth = True
            st.session_state.auth_tab = "Register"
            st.rerun()

    # --------------------------------------------------------
    # 2. DEDICATED AUTHENTICATION OVERLAY (WHEN TRIGGERED)
    # --------------------------------------------------------
    if st.session_state.show_auth:
        st.markdown("<div style='height: 25px;'></div>", unsafe_allow_html=True)
        col_l, col_m, col_r = st.columns([1, 2.2, 1])
        with col_m:
            st.markdown('<div class="auth-box">', unsafe_allow_html=True)
            render_auth_component(key_prefix="modal")
            st.markdown('</div>', unsafe_allow_html=True)
            st.markdown("<div style='height: 15px;'></div>", unsafe_allow_html=True)
            if st.button("← Return to Landing Page Overview", use_container_width=True, key="return_landing_btn"):
                st.session_state.show_auth = False
                st.rerun()
        st.markdown("<div style='height: 40px;'></div>", unsafe_allow_html=True)
        return

    # --------------------------------------------------------
    # 3. HERO SECTION
    # --------------------------------------------------------
    st.markdown("""
    <div class="hero-wrapper">
        <div class="hero-pill">
            <span>✨</span> NEXT-GENERATION AI COPILOT FOR ENGINEERING LABS
        </div>
        <h1 class="hero-title">
            The Intelligent AI Copilot for<br>
            <span class="gradient-text">Engineering Laboratories</span>
        </h1>
        <p class="hero-subtitle">
            Master hands-on experiments, troubleshoot breadboards in real time via computer vision,
            query institutional laboratory manuals with FAISS-grounded RAG, and practice oral vivas with Gemini 3 Flash.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Hero Action Buttons
    h_col1, h_col2, h_col3 = st.columns([2.5, 3.5, 2.5])
    with h_col2:
        btn_c1, btn_c2 = st.columns(2)
        with btn_c1:
            if st.button("🚀 Launch Workspace", key="hero_launch_btn", use_container_width=True):
                st.session_state.show_auth = True
                st.rerun()
        with btn_c2:
            if st.button("⚡ Public Demo (1-Click)", key="hero_guest_btn", use_container_width=True):
                st.session_state.logged_in = True
                st.session_state.user_role = "public"
                st.session_state.user_name = "Guest Explorer"
                st.session_state.user_id = "guest_hero"
                st.rerun()

    # --------------------------------------------------------
    # 4. INTERACTIVE LIVE DIAGNOSTIC SHOWCASE (DUAL-PANE WORKBENCH)
    # --------------------------------------------------------
    st.markdown("""
    <div class="showcase-window">
        <div class="window-header">
            <div class="window-dots">
                <span class="dot dot-red"></span>
                <span class="dot dot-yellow"></span>
                <span class="dot dot-green"></span>
            </div>
            <div class="window-title">labai-diagnostic-engine // live-session: workbench-01.local</div>
            <div style="font-size: 0.75rem; color: #34D399; font-weight: 600;">● INFERENCE ACTIVE</div>
        </div>
        <div class="showcase-body">
            <!-- Pane 1: Hardware Vision Diagnosis -->
            <div class="demo-pane">
                <span class="pane-badge pane-badge-vision">📷 MULTIMODAL COMPUTER VISION INSPECTION</span>
                <p><span class="code-muted">> Target Circuit:</span> <span class="code-cyan">Active Band Pass Filter (LM741)</span></p>
                <p><span class="code-muted">> Input Readings:</span> <span class="code-cyan">Vin = 2.5V pk-pk | f = 10 kHz | Vout = 0.38V</span></p>
                <p><span class="code-muted">> Theoretical Expectation:</span> <span class="code-green">Vout = 1.77V (Passband Peak)</span></p>
                <hr style="border: 0; border-top: 1px solid rgba(255,255,255,0.08); margin: 10px 0;">
                <p><span class="code-amber">⚠️ DETECTED FAULT:</span> <span class="code-red">R2 (10kΩ Feedback) Miswired to Ground</span></p>
                <p><span class="code-muted">> Confidence Score:</span> <span class="code-green">98.4% Confidence</span></p>
                <p><span class="code-muted">> Actionable Fix:</span> Transfer jumper wire from Rail 14B to Pin 2 (Inverting Input). Ensure capacitor C1 polarity matches rail orientation.</p>
            </div>
            <!-- Pane 2: Deterministic RAG Manual Search -->
            <div class="demo-pane">
                <span class="pane-badge pane-badge-rag">📚 DETERMINISTIC FAISS RAG RETRIEVAL</span>
                <p><span class="code-muted">> Student Query:</span> <span class="code-cyan">"How to calculate modulation index for FM wave?"</span></p>
                <p><span class="code-muted">> Syllabus Match:</span> <span class="code-green">Communication Systems Manual • Exp 2 (Page 14)</span></p>
                <p><span class="code-muted">> Canonical Formula:</span> <span class="code-cyan">β = Δf / fm = (f_max - f_min) / (2 × fm)</span></p>
                <hr style="border: 0; border-top: 1px solid rgba(255,255,255,0.08); margin: 10px 0;">
                <p><span class="code-green">✓ MANUAL GROUNDED:</span> Zero hallucination citation verified against institutional laboratory manual corpus.</p>
                <p><span class="code-muted">> Viva Tip:</span> If β &lt; 1, signal is Narrowband FM (NBFM); if β &gt;&gt; 1, Wideband FM (WBFM).</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # --------------------------------------------------------
    # 5. METRICS & IMPACT STRIP
    # --------------------------------------------------------
    m1, m2, m3, m4 = st.columns(4)

    with m1:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-val gradient-text">7+</div>
            <div class="metric-lbl">Lab Experiments</div>
            <div style="font-size: 0.78rem; color: #64748B; margin-top: 4px;">Analog & Digital Modules</div>
        </div>
        """, unsafe_allow_html=True)

    with m2:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-val" style="color: #38BDF8;">35+</div>
            <div class="metric-lbl">Dense RAG Vectors</div>
            <div style="font-size: 0.78rem; color: #64748B; margin-top: 4px;">FAISS Semantic Index</div>
        </div>
        """, unsafe_allow_html=True)

    with m3:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-val" style="color: #34D399;">&lt; 800ms</div>
            <div class="metric-lbl">Diagnostic Latency</div>
            <div style="font-size: 0.78rem; color: #64748B; margin-top: 4px;">Gemini 3 Flash Inference</div>
        </div>
        """, unsafe_allow_html=True)

    with m4:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-val gradient-text-gold">100%</div>
            <div class="metric-lbl">Syllabus Grounded</div>
            <div style="font-size: 0.78rem; color: #64748B; margin-top: 4px;">Zero-Hallucination Citations</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height: 48px;'></div>", unsafe_allow_html=True)

    # --------------------------------------------------------
    # 6. CORE CAPABILITIES (6-PILLAR FEATURE MATRIX)
    # --------------------------------------------------------
    st.markdown("""
    <div style="text-align: center; margin-bottom: 32px;">
        <span class="feature-tag">CUTTING-EDGE ARCHITECTURE</span>
        <h2 style="font-size: 2.3rem; font-weight: 800; color: #FFFFFF; margin-top: 10px;">
            Built for Real Engineering Workbenches
        </h2>
        <p style="color: #94A3B8; max-width: 680px; margin: 0 auto; font-size: 1rem;">
            From hardware circuit diagnosis to rigorous viva voce exam preparation, LabAI empowers every stage of the lab lifecycle.
        </p>
    </div>
    """, unsafe_allow_html=True)

    f_col1, f_col2, f_col3 = st.columns(3)

    with f_col1:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon-wrapper">👁️</div>
            <div class="feature-title">Multimodal Circuit Vision</div>
            <div class="feature-desc">
                Upload photos of physical breadboards or schematics. Gemini 3 Flash detects inverted diodes, loose jumper rails, open ground lines, and wrong resistor color codes.
            </div>
            <span class="feature-tag">Computer Vision + Gemini</span>
        </div>
        """, unsafe_allow_html=True)

    with f_col2:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon-wrapper">📖</div>
            <div class="feature-title">Deterministic Manual RAG</div>
            <div class="feature-desc">
                Never guess pinouts or apparatus specs. Our FAISS semantic retriever indexes institutional lab manuals to extract exact equations, procedures, and component lists.
            </div>
            <span class="feature-tag">FAISS + MiniLM Embeddings</span>
        </div>
        """, unsafe_allow_html=True)

    with f_col3:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon-wrapper">🎓</div>
            <div class="feature-title">AI Viva Voce Sparring</div>
            <div class="feature-desc">
                Rehearse oral examinations before stepping into the lab. Get quizzed on cut-off frequencies, sampling criteria, modulation indices, and circuit topology.
            </div>
            <span class="feature-tag">Oral Exam Simulator</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)

    f_col4, f_col5, f_col6 = st.columns(3)

    with f_col4:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon-wrapper">📊</div>
            <div class="feature-title">Observed vs Theoretical</div>
            <div class="feature-desc">
                Enter your oscilloscope readings (Vin, Vout, f). LabAI instantly calculates percentage deviations, validates passbands, and flags anomalous lab findings.
            </div>
            <span class="feature-tag">Automated Verification</span>
        </div>
        """, unsafe_allow_html=True)

    with f_col5:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon-wrapper">🛡️</div>
            <div class="feature-title">Role-Based Lab Portals</div>
            <div class="feature-desc">
                Tailored workflows for Students (guided self-service troubleshooting), Lab Technicians (fault logging & verification), and Public Visitors (open exploratory access).
            </div>
            <span class="feature-tag">Multi-Tenant RBAC</span>
        </div>
        """, unsafe_allow_html=True)

    with f_col6:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon-wrapper">⚡</div>
            <div class="feature-title">Zero-Latency Vector Search</div>
            <div class="feature-desc">
                High-performance offline FAISS vector engine running dense 384-dimensional embeddings directly within the host environment for sub-second query retrieval.
            </div>
            <span class="feature-tag">Local FAISS Vectors</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height: 52px;'></div>", unsafe_allow_html=True)

    # --------------------------------------------------------
    # 7. WORKFLOW: HOW IT WORKS (3-STEP PIPELINE)
    # --------------------------------------------------------
    st.markdown("""
    <div style="text-align: center; margin-bottom: 32px;">
        <span class="feature-tag">SEAMLESS WORKFLOW</span>
        <h2 style="font-size: 2.3rem; font-weight: 800; color: #FFFFFF; margin-top: 10px;">
            How LabAI Works in Your Lab
        </h2>
    </div>
    """, unsafe_allow_html=True)

    w1, w2, w3 = st.columns(3)

    with w1:
        st.markdown("""
        <div class="step-card">
            <div class="step-num">01</div>
            <div style="font-weight: 700; font-size: 1.15rem; color: #FFFFFF; margin-bottom: 8px;">Capture or Inquire</div>
            <div style="color: #94A3B8; font-size: 0.92rem; line-height: 1.6;">
                Snap a photo of your breadboard circuit or type any procedural doubt (e.g., "What is the procedure for Flat Top Sampling?").
            </div>
        </div>
        """, unsafe_allow_html=True)

    with w2:
        st.markdown("""
        <div class="step-card">
            <div class="step-num">02</div>
            <div style="font-weight: 700; font-size: 1.15rem; color: #FFFFFF; margin-bottom: 8px;">Multimodal Reasoning & RAG</div>
            <div style="color: #94A3B8; font-size: 0.92rem; line-height: 1.6;">
                FAISS vector retriever searches the official lab manual while Gemini 3 Flash performs cross-modal computer vision inspection.
            </div>
        </div>
        """, unsafe_allow_html=True)

    with w3:
        st.markdown("""
        <div class="step-card">
            <div class="step-num">03</div>
            <div style="font-weight: 700; font-size: 1.15rem; color: #FFFFFF; margin-bottom: 8px;">Surgical Fix & Viva Ready</div>
            <div style="color: #94A3B8; font-size: 0.92rem; line-height: 1.6;">
                Receive pinpointed circuit corrections, step-by-step procedure sequences, exact manual page citations, and viva answers.
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height: 52px;'></div>", unsafe_allow_html=True)

    # --------------------------------------------------------
    # 8. SUPPORTED CURRICULUM DIRECTORY
    # --------------------------------------------------------
    st.markdown("""
    <div style="text-align: center; margin-bottom: 30px;">
        <span class="feature-tag">COMPREHENSIVE SYLLABUS</span>
        <h2 style="font-size: 2.3rem; font-weight: 800; color: #FFFFFF; margin-top: 10px;">
            Supported Laboratory Modules
        </h2>
    </div>
    """, unsafe_allow_html=True)

    curr_col1, curr_col2 = st.columns(2)

    with curr_col1:
        st.markdown("""
        <div class="feature-card">
            <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 14px;">
                <span style="font-weight: 800; font-size: 1.25rem; color: #FFFFFF;">📡 Communication Systems Lab</span>
                <span class="status-pill" style="background: rgba(99,102,241,0.15); color: #818CF8; border-color: rgba(99,102,241,0.3);">7 Experiments</span>
            </div>
            <p style="color: #94A3B8; font-size: 0.9rem; margin-bottom: 16px;">
                Full RAG-indexed institutional lab manual with step-by-step procedures, circuit schematics, formulas, and viva questions.
            </p>
            <div style="display: flex; flex-direction: column; gap: 8px;">
                <div style="background: rgba(255,255,255,0.04); padding: 8px 12px; border-radius: 8px; font-size: 0.85rem; color: #E2E8F0;">
                    🔹 <b>Exp 1:</b> Amplitude Modulation & Demodulation (AM)
                </div>
                <div style="background: rgba(255,255,255,0.04); padding: 8px 12px; border-radius: 8px; font-size: 0.85rem; color: #E2E8F0;">
                    🔹 <b>Exp 2:</b> Frequency Modulation & Detection (FM)
                </div>
                <div style="background: rgba(255,255,255,0.04); padding: 8px 12px; border-radius: 8px; font-size: 0.85rem; color: #E2E8F0;">
                    🔹 <b>Exp 3:</b> Pulse Amplitude Modulation & Demodulation (PAM)
                </div>
                <div style="background: rgba(255,255,255,0.04); padding: 8px 12px; border-radius: 8px; font-size: 0.85rem; color: #E2E8F0;">
                    🔹 <b>Exp 4:</b> Pulse Width Modulation (PWM)
                </div>
                <div style="background: rgba(255,255,255,0.04); padding: 8px 12px; border-radius: 8px; font-size: 0.85rem; color: #E2E8F0;">
                    🔹 <b>Exp 5:</b> Flat Top Sampling & Reconstruction
                </div>
                <div style="background: rgba(255,255,255,0.04); padding: 8px 12px; border-radius: 8px; font-size: 0.85rem; color: #E2E8F0;">
                    🔹 <b>Exp 6 & 7:</b> Amplitude Shift Keying (ASK) & Phase Shift Keying (PSK)
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with curr_col2:
        st.markdown("""
        <div class="feature-card">
            <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 14px;">
                <span style="font-weight: 800; font-size: 1.25rem; color: #FFFFFF;">🔌 Electronics & Circuit Diagnosis</span>
                <span class="status-pill" style="background: rgba(56,189,248,0.15); color: #38BDF8; border-color: rgba(56,189,248,0.3);">Multimodal AI</span>
            </div>
            <p style="color: #94A3B8; font-size: 0.9rem; margin-bottom: 16px;">
                Real-time breadboard image analysis coupled with observed voltage/frequency measurements for instant hardware fault isolation.
            </p>
            <div style="display: flex; flex-direction: column; gap: 8px;">
                <div style="background: rgba(255,255,255,0.04); padding: 8px 12px; border-radius: 8px; font-size: 0.85rem; color: #E2E8F0;">
                    ⚡ <b>Band Pass Filter:</b> Resonant frequency f0 = 1 / (2π√(R1R2C1C2)), bandwidth & Q-factor debugging
                </div>
                <div style="background: rgba(255,255,255,0.04); padding: 8px 12px; border-radius: 8px; font-size: 0.85rem; color: #E2E8F0;">
                    ⚡ <b>Precision Voltage Divider:</b> Vout = Vin × (R2 / (R1 + R2)), load resistance & tolerance inspection
                </div>
                <div style="background: rgba(255,255,255,0.04); padding: 8px 12px; border-radius: 8px; font-size: 0.85rem; color: #E2E8F0;">
                    ⚡ <b>LED Driver Circuit:</b> Current limiting resistor validation, polarity checks & forward bias analysis
                </div>
                <div style="background: rgba(99,102,241,0.08); padding: 14px; border-radius: 8px; font-size: 0.85rem; color: #A5B4FC; margin-top: 6px; border: 1px dashed rgba(99,102,241,0.3);">
                    ✨ <b>Multimodal Fault Dataset:</b> Cross-referenced against real-world hardware failure patterns & reference lab cases.
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height: 52px;'></div>", unsafe_allow_html=True)

    # --------------------------------------------------------
    # 9. TRADITIONAL VS LABAI COMPARISON
    # --------------------------------------------------------
    st.markdown("""
    <div style="text-align: center; margin-bottom: 30px;">
        <span class="feature-tag">WHY ENGINEERING LABS CHOOSE LABAI</span>
        <h2 style="font-size: 2.3rem; font-weight: 800; color: #FFFFFF; margin-top: 10px;">
            The Modern Lab Transformation
        </h2>
    </div>
    """, unsafe_allow_html=True)

    c_left, c_right = st.columns(2)

    with c_left:
        st.markdown("""
        <div class="comp-card comp-bad">
            <div class="comp-title" style="color: #F87171;">
                ❌ Traditional Lab Experience
            </div>
            <div class="comp-item">❌ <b>Lengthy TA Delays:</b> Students wait 20+ minutes for instructor assistance with basic wiring mistakes.</div>
            <div class="comp-item">❌ <b>Damaged Components:</b> Burnt ICs and blown transistors caused by unverified power connections.</div>
            <div class="comp-item">❌ <b>Manual Paper Search:</b> Struggling to locate correct formulas or pinouts in dog-eared physical manuals.</div>
            <div class="comp-item">❌ <b>Unprepared Oral Vivas:</b> High anxiety and low confidence due to zero mock oral examination practice.</div>
        </div>
        """, unsafe_allow_html=True)

    with c_right:
        st.markdown("""
        <div class="comp-card comp-good">
            <div class="comp-title" style="color: #34D399;">
                ✅ With LabAI Copilot
            </div>
            <div class="comp-item">✅ <b>Instant Multimodal Diagnosis:</b> Snap a photo and get exact miswired pins identified in &lt; 800ms.</div>
            <div class="comp-item">✅ <b>Proactive Safe Verification:</b> Verify circuit connections before turning on the DC power supply.</div>
            <div class="comp-item">✅ <b>Deterministic Manual RAG:</b> Canonical formulas and procedures retrieved instantly with page citations.</div>
            <div class="comp-item">✅ <b>AI Viva Sparring Partner:</b> Dynamic oral questions and conceptual scoring for top exam performance.</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height: 60px;'></div>", unsafe_allow_html=True)

    # --------------------------------------------------------
    # 10. FAST ACCESS & AUTHENTICATION PORTAL (BOTTOM SECTION)
    # --------------------------------------------------------
    st.markdown("""
    <div style="text-align: center; margin-bottom: 24px;">
        <span class="feature-tag">GET STARTED TODAY</span>
        <h2 style="font-size: 2.3rem; font-weight: 800; color: #FFFFFF; margin-top: 10px;">
            Ready to Accelerate Your Lab Experiments?
        </h2>
        <p style="color: #94A3B8; font-size: 1rem;">
            Choose your access mode below to enter the interactive workspace.
        </p>
    </div>
    """, unsafe_allow_html=True)

    cta_l, cta_m, cta_r = st.columns([1, 2.4, 1])
    with cta_m:
        st.markdown('<div class="auth-box">', unsafe_allow_html=True)
        render_auth_component(key_prefix="bottom")
        st.markdown('</div>', unsafe_allow_html=True)

    # --------------------------------------------------------
    # 11. PROFESSIONAL FOOTER
    # --------------------------------------------------------
    st.markdown("""
    <div class="footer-wrap">
        <div class="footer-badges">
            <span class="tech-badge">⚡ Google Gemini 3 Flash</span>
            <span class="tech-badge">🔍 FAISS Vector Database</span>
            <span class="tech-badge">🧠 SentenceTransformers</span>
            <span class="tech-badge">🍃 MongoDB Atlas</span>
            <span class="tech-badge">🚀 Streamlit</span>
        </div>
        <p style="margin: 0 0 6px 0; color: #94A3B8; font-weight: 600;">
            LabAI — Advanced Multimodal Laboratory Intelligence Platform
        </p>
        <p style="margin: 0; font-size: 0.8rem; color: #64748B;">
            Designed for Electronics & Communication Engineering Laboratories • Institutional Syllabus Grounded
        </p>
    </div>
    """, unsafe_allow_html=True)


# ============================================================
# FRONTEND COMPONENTS — presentation only; backend calls unchanged
# ============================================================

st.markdown("""
<style>
.block-container { max-width: 1180px !important; }
.nav-wordmark { font-size: 1.35rem; font-weight: 800; letter-spacing: -.04em; padding: .55rem 0; color: #F8FAFC; }
.nav-wordmark span { color: #818CF8; } .nav-links { color: #CBD5E1; font-size: .9rem; text-align: center; padding: .7rem 0; white-space: nowrap; }
.hero-clean { text-align:center; padding: 7.5rem 1rem 3rem; } .eyebrow,.section-heading>span,.cta-clean>span { color:#A5B4FC; font-size:.72rem; font-weight:800; letter-spacing:.13em; }
.hero-clean h1 { color:#F8FAFC; font-size:clamp(2.7rem,6vw,5.2rem); letter-spacing:-.07em; line-height:1.02; margin:1rem 0 1.35rem; }
.hero-clean h1 span { background:linear-gradient(100deg,#A5B4FC,#60A5FA,#C4B5FD); -webkit-background-clip:text; -webkit-text-fill-color:transparent; }
.hero-clean p { color:#CBD5E1; max-width:720px; font-size:1.12rem; line-height:1.7; margin:auto; }
.metric-card.clean { margin: 4.5rem 0; padding:1.4rem; text-align:left; } .metric-card.clean strong { display:block; font-size:1.7rem; color:#F8FAFC; } .metric-card.clean span { color:#94A3B8; font-size:.85rem; }
.section-heading { text-align:center; margin:6rem auto 2.5rem; } .section-heading h2,.cta-clean h2 { color:#F8FAFC; font-size:clamp(2rem,4vw,3rem); letter-spacing:-.05em; margin:.7rem 0; } .section-heading p { color:#94A3B8; }
.feature-card.clean { min-height:210px; margin-bottom:1.25rem; } .feature-card.clean i { display:block; color:#93C5FD; font-style:normal; font-size:1.8rem; margin-bottom:1rem; } .feature-card.clean h3,.workflow-card h3,.roadmap-card h3 { color:#F8FAFC; margin:0 0 .7rem; } .feature-card.clean p,.workflow-card p,.comparison p { color:#CBD5E1; line-height:1.65; }
.workflow-card,.comparison,.roadmap-card { border:1px solid rgba(148,163,184,.16); background:rgba(30,41,59,.56); border-radius:18px; padding:1.7rem; min-height:190px; margin-bottom:1rem; } .workflow-card small { color:#818CF8; font-weight:800; } .comparison.labai { border-color:rgba(96,165,250,.38); background:rgba(30,41,59,.8); } .comparison h3 { color:#F8FAFC; } .roadmap-card { min-height:120px; } .roadmap-card span { font-size:.67rem; color:#94A3B8; letter-spacing:.1em; }
.cta-clean { text-align:center; margin:6rem 0 1.5rem; padding:4rem 1rem; background:radial-gradient(circle,rgba(99,102,241,.24),transparent 65%); } .cta-clean p { color:#CBD5E1; }
.auth-shell { text-align:center; margin-top:4rem; } .auth-shell h2 { color:#F8FAFC; margin:.6rem 0; } .auth-shell p,.field-label { color:#CBD5E1; } @media(max-width:800px){ .nav-links{display:none}.hero-clean{padding-top:4rem}.metric-card.clean{margin:2rem 0;} }
.dashboard-hero { display:flex; justify-content:space-between; align-items:flex-start; gap:1rem; padding:2.2rem; margin:1rem 0 2rem; border:1px solid rgba(129,140,248,.25); border-radius:20px; background:linear-gradient(120deg,rgba(99,102,241,.2),rgba(30,41,59,.72)); } .dashboard-hero span,.surface-card>span,.chat-assistant span { color:#A5B4FC; font-size:.7rem; letter-spacing:.12em; font-weight:800; } .dashboard-hero h1 { margin:.5rem 0; color:#F8FAFC; font-size:clamp(2rem,4vw,3.5rem); letter-spacing:-.05em; } .dashboard-hero p,.surface-card p { color:#CBD5E1; } .role-badge { display:inline-block; padding:.38rem .7rem; border:1px solid rgba(129,140,248,.35); border-radius:999px; color:#C7D2FE; background:rgba(99,102,241,.14); font-size:.76rem; font-weight:700; }
.surface-card { background:rgba(30,41,59,.6); border:1px solid rgba(148,163,184,.15); border-radius:16px; padding:1.35rem; margin:1rem 0; } .surface-card h2,.surface-card h3 { color:#F8FAFC; margin:.45rem 0; } .chat-assistant { margin-top:.5rem; } .dashboard-card { min-height:180px; }
[data-testid="stMarkdownContainer"] > p, [data-testid="stMarkdownContainer"] > ul, [data-testid="stMarkdownContainer"] > ol, [data-testid="stMarkdownContainer"] > h1, [data-testid="stMarkdownContainer"] > h2, [data-testid="stMarkdownContainer"] > h3, [data-testid="stMarkdownContainer"] > h4, [data-testid="stMarkdownContainer"] > h5, [data-testid="stMarkdownContainer"] > h6, [data-testid="stMarkdownContainer"] > ul li, [data-testid="stMarkdownContainer"] > ol li { color:#FFFFFF !important; }
[data-testid="stMarkdownContainer"] > h1, [data-testid="stMarkdownContainer"] > h2, [data-testid="stMarkdownContainer"] > h3, [data-testid="stMarkdownContainer"] > h4, [data-testid="stMarkdownContainer"] > h5, [data-testid="stMarkdownContainer"] > h6 { color:#FFFFFF !important; font-weight:800 !important; }
[data-testid="stExpander"] summary, [data-testid="stExpander"] summary p, [data-testid="stExpander"] [data-testid="stMarkdownContainer"], [data-testid="stExpander"] [data-testid="stMarkdownContainer"] p, [data-testid="stExpander"] [data-testid="stMarkdownContainer"] li, [data-testid="stExpander"] [data-testid="stMarkdownContainer"] h1, [data-testid="stExpander"] [data-testid="stMarkdownContainer"] h2, [data-testid="stExpander"] [data-testid="stMarkdownContainer"] h3, [data-testid="stExpander"] [data-testid="stMarkdownContainer"] h4 { color:#FFFFFF !important; }
[data-testid="stExpander"] { border-color:rgba(148,163,184,.25) !important; }
[data-testid="stSidebar"], [data-testid="stSidebar"] [data-testid="stMarkdownContainer"], [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p, [data-testid="stSidebar"] label, [data-testid="stSidebar"] label p, [data-testid="stSidebar"] [data-testid="stCaptionContainer"], [data-testid="stSidebar"] [data-testid="stCaptionContainer"] p, [data-testid="stSidebar"] .stButton > button { color:#000000 !important; }
[data-testid="stSidebar"] .role-badge { color:#000000 !important; }
[data-testid="stSidebar"] [role="radiogroup"] { gap: .35rem !important; }
[data-testid="stSidebar"] [role="radiogroup"] label { display:flex !important; align-items:center !important; width:100% !important; padding:.65rem .8rem !important; border-radius:10px !important; cursor:pointer !important; transition:background .2s ease, color .2s ease !important; }
[data-testid="stSidebar"] [role="radiogroup"] label:hover { background:rgba(99,102,241,.12) !important; }
[data-testid="stSidebar"] [role="radiogroup"] label > div:first-child { display:none !important; }
[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) { background:#4F46E5 !important; color:#FFFFFF !important; box-shadow:0 4px 14px rgba(79,70,229,.25) !important; }
[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) p { color:#FFFFFF !important; font-weight:700 !important; }
</style>""", unsafe_allow_html=True)

def _open_auth(mode="Login"):
    st.session_state.show_auth = True
    st.session_state.auth_tab = mode


def render_navbar():
    brand, links, login, start = st.columns([2.2, 4.2, 1.1, 1.35])
    with brand:
        st.markdown('<div class="nav-wordmark"><span>✦</span> LabAI</div>', unsafe_allow_html=True)
    with links:
        st.markdown('<div class="nav-links">Features&nbsp;&nbsp;&nbsp; Experiments&nbsp;&nbsp;&nbsp; Research Hub&nbsp;&nbsp;&nbsp; About</div>', unsafe_allow_html=True)
    with login:
        if st.button("Login", key="nav_login", use_container_width=True):
            _open_auth("Login")
            st.rerun()
    with start:
        if st.button("Get Started", key="nav_start", use_container_width=True):
            _open_auth("Register")
            st.rerun()


def render_hero():
    st.markdown('''<section class="hero-clean">
      <div class="eyebrow">LABORATORY INTELLIGENCE, GROUNDED IN YOUR MANUAL</div>
      <h1>The AI Copilot for<br><span>Engineering Laboratories</span></h1>
      <p>Master experiments, prepare for vivas, troubleshoot faults, and learn engineering concepts using Retrieval-Augmented AI.</p>
    </section>''', unsafe_allow_html=True)
    left, middle, right = st.columns([2.6, 2.8, 2.6])
    with middle:
        a, b = st.columns(2)
        with a:
            if st.button("Get Started", key="hero_get_started", use_container_width=True):
                _open_auth("Register")
                st.rerun()
        with b:
            if st.button("Explore Platform", key="hero_explore", use_container_width=True):
                _open_auth("Login")
                st.rerun()


def render_statistics():
    metrics = [("7+", "Experiments"), ("35+", "Knowledge Chunks"), ("24/7", "AI Assistance"), ("Role-Based", "Access Control")]
    cols = st.columns(4)
    for col, (value, label) in zip(cols, metrics):
        with col:
            st.markdown(f'<div class="metric-card clean"><strong>{value}</strong><span>{label}</span></div>', unsafe_allow_html=True)


def render_features():
    st.markdown('<div class="section-heading"><span>CAPABILITIES</span><h2>Everything your lab session needs</h2><p>One focused workspace for practical learning, preparation, and problem solving.</p></div>', unsafe_allow_html=True)
    cards = [("◌", "Experiment Intelligence", "Structured procedures and theory drawn from the laboratory manual."),
             ("✦", "AI Laboratory Tutor", "Ask focused questions and receive grounded explanations."),
             ("◇", "Viva Preparation", "Build confidence with concepts, formulas, and oral-exam prompts."),
             ("⌁", "Fault Diagnosis", "Pair circuit images and measurements with practical next steps."),
             ("↗", "Research Hub", "Explore institutional knowledge with semantic retrieval."),
             ("▤", "Knowledge Repository", "Keep experiment context, procedures, and sources together.")]
    for row in (cards[:3], cards[3:]):
        cols = st.columns(3)
        for col, (icon, title, text) in zip(cols, row):
            with col:
                st.markdown(f'<article class="feature-card clean"><i>{icon}</i><h3>{title}</h3><p>{text}</p></article>', unsafe_allow_html=True)


def render_how_it_works():
    st.markdown('<div class="section-heading"><span>HOW IT WORKS</span><h2>Grounded answers, not guesses</h2></div>', unsafe_allow_html=True)
    steps = [("01", "Knowledge Sources", "Your laboratory manual and curated experiment data."), ("02", "FAISS Retrieval", "Relevant context is located before an answer is generated."), ("03", "Gemini Grounded Response", "A clear response is composed from the retrieved evidence.")]
    cols = st.columns(3)
    for col, (number, title, text) in zip(cols, steps):
        with col:
            st.markdown(f'<div class="workflow-card"><small>{number}</small><h3>{title}</h3><p>{text}</p></div>', unsafe_allow_html=True)


def render_why_labai():
    st.markdown('<div class="section-heading"><span>WHY LABAI</span><h2>Designed for the reality of lab learning</h2></div>', unsafe_allow_html=True)
    old, new = st.columns(2)
    with old:
        st.markdown('<div class="comparison traditional"><h3>Traditional Learning</h3><p>Generic answers without experiment context</p><p>Manual searching under time pressure</p><p>Limited support outside lab hours</p></div>', unsafe_allow_html=True)
    with new:
        st.markdown('<div class="comparison labai"><h3>LabAI</h3><p>✓ Grounded responses</p><p>✓ Reduced hallucination</p><p>✓ Experiment context awareness</p><p>✓ Institution-specific knowledge</p></div>', unsafe_allow_html=True)


def render_future_vision():
    st.markdown('<div class="section-heading"><span>FUTURE VISION</span><h2>One platform for engineering knowledge</h2></div>', unsafe_allow_html=True)
    roadmap = ["Research Paper Learning", "DSP Learning Assistant", "VLSI Learning Assistant", "Embedded Systems Assistant", "Wireless Communications Assistant", "Engineering Knowledge Platform"]
    cols = st.columns(3)
    for idx, item in enumerate(roadmap):
        with cols[idx % 3]:
            st.markdown(f'<div class="roadmap-card"><span>COMING NEXT</span><h3>{item}</h3></div>', unsafe_allow_html=True)


def render_cta():
    st.markdown('<section class="cta-clean"><span>START LEARNING WITH CONTEXT</span><h2>Ready to Transform Laboratory Learning?</h2><p>Enter a workspace built around the way engineering students actually learn.</p></section>', unsafe_allow_html=True)
    _, center, _ = st.columns([2.5, 3, 2.5])
    with center:
        a, b = st.columns(2)
        with a:
            if st.button("Get Started", key="cta_get_started", use_container_width=True):
                _open_auth("Register")
                st.rerun()
        with b:
            if st.button("Create Account", key="cta_create", use_container_width=True):
                _open_auth("Register")
                st.rerun()


def render_authentication():
    _, card, _ = st.columns([1.4, 2, 1.4])
    with card:
        st.markdown('<div class="auth-shell"><div class="eyebrow">LABAI ACCESS</div><h2>Enter your workspace</h2><p>Choose how you would like to continue.</p></div>', unsafe_allow_html=True)
        login_col, register_col = st.columns(2)
        with login_col:
            if st.button("Login", key="auth_mode_login", use_container_width=True):
                st.session_state.auth_tab = "Login"
        with register_col:
            if st.button("Register", key="auth_mode_register", use_container_width=True):
                st.session_state.auth_tab = "Register"
        if st.session_state.auth_tab == "Login":
            email = st.text_input("Email", key="clean_login_email")
            password = st.text_input("Password", type="password", key="clean_login_password")
            if st.button("Login to LabAI", key="clean_login_submit", use_container_width=True):
                if not email or not password:
                    st.warning("Please provide both email and password.")
                else:
                    with st.spinner("Authenticating..."):
                        success, user = login_user(email, password)
                    if success:
                        st.session_state.logged_in = True
                        st.session_state.user_name = user.get("name", "Student")
                        st.session_state.user_role = user.get("role", "student")
                        st.session_state.user_id = str(user.get("_id", ""))
                        st.session_state.show_auth = False
                        st.rerun()
                    st.error("Invalid credentials. Check your email and password.") if not success else None
        else:
            st.markdown('<p class="field-label">Select your role</p>', unsafe_allow_html=True)
            roles = [("Student", "student"), ("Lab Technician", "technician"), ("Public Repository", "public")]
            role_cols = st.columns(3)
            for col, (label, value) in zip(role_cols, roles):
                with col:
                    if st.button(label, key=f"role_{value}", use_container_width=True):
                        st.session_state.register_role = value
            role = st.session_state.get("register_role", "student")
            st.caption(f"Selected role: {role.replace('_', ' ').title()}")
            name = st.text_input("Full name", key="clean_register_name")
            email = st.text_input("Email", key="clean_register_email")
            password = st.text_input("Password", type="password", key="clean_register_password")
            if st.button("Create account", key="clean_register_submit", use_container_width=True):
                if not name or not email or not password:
                    st.warning("Please complete all registration fields.")
                else:
                    with st.spinner("Creating account..."):
                        success, message = register_user(name, email, password, role)
                    st.success("Account created successfully. Please log in.") if success else st.error(message)
        st.divider()
        if st.button("Explore Public Repository", key="clean_guest", use_container_width=True):
            st.session_state.logged_in, st.session_state.user_role = True, "public"
            st.session_state.user_name, st.session_state.user_id = "Guest Explorer", "guest_temp"
            st.session_state.show_auth = False
            st.rerun()
        if st.button("← Back to LabAI", key="clean_auth_close", use_container_width=True):
            st.session_state.show_auth = False
            st.rerun()


def show_landing_page():
    render_navbar()
    if st.session_state.show_auth:
        render_authentication()
        return
    render_hero()
    render_statistics()
    render_features()
    render_how_it_works()
    render_why_labai()
    render_future_vision()
    render_cta()


# ============================================================
# APP ROUTING & ENTRYPOINT
# ============================================================

if not st.session_state.logged_in:
    show_landing_page()
    st.stop()

# ============================================================
# LOAD EXISTING KNOWLEDGE BASE
# ============================================================

try:

    with open("knowledge_base.json", "r", encoding="utf-8") as file:
        knowledge_base = json.load(file)

except Exception as e:

    st.error(f"Could not load knowledge_base.json: {e}")
    st.stop()


# ============================================================
# LOAD EXISTING FAULT DATASET
# ============================================================

try:

    with open("fault_dataset.json", "r", encoding="utf-8") as file:
        fault_dataset = json.load(file)

except Exception as e:

    st.error(f"Could not load fault_dataset.json: {e}")
    st.stop()


# ============================================================
# LOAD LAB RAG DATABASES
# ============================================================

LAB_DATABASE_PATHS = {
    "Communication Systems": {
        "index": os.path.join("experiments", "combined_database", "faiss.index"),
        "chunks": os.path.join("experiments", "combined_database", "chunks.json")
    },
    "Analog Electronics Circuits": {
        "index": os.path.join("aec", "aec_faiss.index"),
        "chunks": os.path.join("aec", "aec_chunks.json")
    }
}


def load_lab_database(lab_name):
    """Load the FAISS index and chunk records for a supported laboratory."""
    paths = LAB_DATABASE_PATHS.get(lab_name)
    if paths is None:
        raise ValueError(f"Unsupported laboratory: {lab_name}")

    faiss_index = faiss.read_index(paths["index"])
    with open(paths["chunks"], "r", encoding="utf-8") as file:
        rag_chunks = json.load(file)
    return faiss_index, rag_chunks


try:
    lab_databases = {
        lab_name: load_lab_database(lab_name)
        for lab_name in LAB_DATABASE_PATHS
    }

    faiss_index, rag_chunks = lab_databases["Communication Systems"]

    with open(
        os.path.join("experiments", "combined_database", "procedure_database.json"),
        "r",
        encoding="utf-8"
    ) as file:
        procedure_database = json.load(file)

    embedding_model = SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2",
        local_files_only=True
    )

except Exception as e:
    st.error("Laboratory RAG database could not be loaded.")
    st.error(str(e))
    st.stop()


AEC_PDF_PATH = os.path.join("aec", "AEC.pdf")


# ============================================================
# EXPERIMENT INFORMATION
# ============================================================

# The learning repository follows the procedure database, so new experiments
# appear automatically when they are added to procedure_database.json.
communication_experiments = {
    str(number): {
        "name": procedure.get("experiment_name", f"Experiment {number}"),
        "short_name": procedure.get("short_name", "")
    }
    for number, procedure in procedure_database.items()
}


AEC_REPOSITORY_EXPERIMENTS = [
    {"number": 1, "name": "Regulated DC Power Supply", "pages": (5, 7), "summary": "Design and test a regulated DC power supply with rectifier, filter, and regulator stages."},
    {"number": 2, "name": "Class B Push-Pull Power Amplifier", "pages": (8, 10), "summary": "Study push-pull operation, efficiency, and crossover distortion."},
    {"number": 3, "name": "4-Bit R-2R Digital-to-Analog Converter", "pages": (11, 14), "summary": "Design an R-2R ladder DAC and verify its binary-to-analog output."},
    {"number": 4, "name": "2-Bit Flash Analog-to-Digital Converter", "pages": (15, 16), "summary": "Design a flash ADC using comparators and interpret its digital output."},
    {"number": 5, "name": "Differential and Instrumentation Amplifier", "pages": (17, 20), "summary": "Measure differential gain, common-mode gain, and common-mode rejection."},
    {"number": 6, "name": "Clippers and Clampers", "pages": (24, 30), "summary": "Design diode clipping and clamping circuits to control waveform levels."},
    {"number": 7, "name": "Precision Rectifier", "pages": (31, 33), "summary": "Design and test a precision rectifier for accurate low-level signal rectification."},
    {"number": 8, "name": "Schmitt Trigger", "pages": (34, 36), "summary": "Design a Schmitt trigger and determine its upper and lower threshold points."},
    {"number": 9, "name": "Astable Multivibrator", "pages": (37, 40), "summary": "Design a 555-timer astable multivibrator and verify its timing waveform."},
    {"number": 10, "name": "Monostable Multivibrator", "pages": (41, 42), "summary": "Design a 555-timer monostable circuit and measure its output pulse width."},
    {"number": 11, "name": "Single-Stage MOSFET Amplifier", "pages": (43, 50), "summary": "Study voltage amplification and frequency response in a MOSFET amplifier."}
]


EXPERIMENT_DIAGNOSTIC_INPUTS = {
    "band_pass_filter": [
        {"key": "low_cutoff", "label": "Low cutoff frequency (Hz)", "help": "Measured lower -3 dB cutoff frequency."},
        {"key": "high_cutoff", "label": "High cutoff frequency (Hz)", "help": "Measured upper -3 dB cutoff frequency."},
        {"key": "center_frequency", "label": "Center frequency (Hz)", "help": "Frequency where the output is maximum."},
        {"key": "resistor_value", "label": "Resistor value (ohms)", "help": "Value printed or measured for the main filter resistor."},
        {"key": "capacitor_value", "label": "Capacitor value (F)", "help": "Value printed or measured for the main filter capacitor."}
    ],
    "voltage_divider": [
        {"key": "input_voltage", "label": "Input voltage Vin (V)", "help": "Voltage applied to the divider."},
        {"key": "output_voltage", "label": "Measured output voltage Vout (V)", "help": "Voltage measured across the load resistor."},
        {"key": "r1_value", "label": "R1 value (ohms)", "help": "Upper resistor value."},
        {"key": "r2_value", "label": "R2 value (ohms)", "help": "Lower resistor or load value."}
    ],
    "led_circuit": [
        {"key": "supply_voltage", "label": "Supply voltage (V)", "help": "Voltage connected to the LED circuit."},
        {"key": "resistor_value", "label": "Current-limiting resistor (ohms)", "help": "Series resistor value."},
        {"key": "led_current", "label": "Measured LED current (mA)", "help": "Current flowing through the LED."},
        {"key": "led_forward_voltage", "label": "LED forward voltage Vf (V)", "help": "Measured or datasheet forward voltage."}
    ]
}


# ============================================================
# REUSABLE RAG SEARCH FUNCTIONS
# ============================================================

def embed_query(query, dimension):
    """Create a normalized query vector matching the target FAISS index."""
    if dimension == embedding_model.get_sentence_embedding_dimension():
        query_embedding = embedding_model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True
        )
    elif dimension == 3072:
        response = client.models.embed_content(
            model="gemini-embedding-001",
            contents=query,
            config=types.EmbedContentConfig(output_dimensionality=dimension)
        )
        query_embedding = np.asarray(
            response.embeddings[0].values,
            dtype="float32"
        ).reshape(1, -1)
        query_embedding /= np.linalg.norm(query_embedding, axis=1, keepdims=True)
    else:
        raise ValueError(f"No query embedder is configured for dimension {dimension}.")

    return query_embedding.astype("float32")


def search_faiss(query, faiss_index, rag_chunks, k=5):

    query_embedding = embed_query(query, faiss_index.d)

    scores, indices = faiss_index.search(
        query_embedding,
        k
    )

    results = []

    for score, index in zip(scores[0], indices[0]):

        if index < 0 or index >= len(rag_chunks):
            continue

        chunk = rag_chunks[int(index)]

        results.append({

            "score": float(score),

            "experiment_number": chunk.get("experiment_number"),

            "experiment": chunk.get(
                "experiment_name",
                "Analog Electronics Circuits"
            ),

            "page": chunk.get("manual_page", chunk.get("page", "Not specified")),

            "text": chunk.get("text", "")

        })

    return results


def retrieve_chunks(query, lab_name="Communication Systems", k=5):
    """Retrieve normalized chunks from the selected laboratory database."""
    selected_index, selected_chunks = lab_databases[lab_name]
    return search_faiss(query, selected_index, selected_chunks, k=k)


def search_lab_manual(query, k=5):
    """Backward-compatible Communication Systems retrieval wrapper."""
    return retrieve_chunks(query, "Communication Systems", k=k)


# ============================================================
# DETECT COMMUNICATION SYSTEMS EXPERIMENT
# ============================================================

def detect_communication_experiment(question):

    question_lower = question.lower()

    keywords = {

        "1": [
            "amplitude modulation",
            "am modulation",
            "am"
        ],

        "2": [
            "frequency modulation",
            "fm modulation"
        ],

        "3": [
            "pulse amplitude modulation",
            "pam modulation",
            "pam"
        ],

        "4": [
            "pulse width modulation",
            "pwm modulation",
            "pwm"
        ],

        "5": [
            "flat top sampling",
            "flat top"
        ],

        "6": [
            "amplitude shift keying",
            "ask modulation",
            "ask"
        ],

        "7": [
            "phase shift keying",
            "psk modulation",
            "psk"
        ]

    }

    # Longer keywords first
    all_keywords = []

    for experiment_number, words in keywords.items():

        for word in words:

            all_keywords.append(
                (word, experiment_number)
            )

    all_keywords.sort(
        key=lambda x: len(x[0]),
        reverse=True
    )

    for word, experiment_number in all_keywords:

        if word in question_lower:

            return experiment_number

    return None


# ============================================================
# PROCEDURE SEARCH
# ============================================================

def get_procedure(experiment_number):

    experiment_number = str(experiment_number)

    if experiment_number in procedure_database:

        return procedure_database[experiment_number]

    return None


def clean_manual_text(text):
    """Make OCR text readable while preserving every source line."""
    cleaned_lines = []
    for line in str(text).splitlines():
        line = re.sub(r"\s+", " ", line).strip()
        if line:
            cleaned_lines.append(line)
    return "\n".join(cleaned_lines)


def split_manual_sections(text):
    """Extract common manual headings without requiring fixed experiment fields."""
    cleaned_text = clean_manual_text(text)
    heading_pattern = re.compile(
        r"(?im)^\s*(aim|objective|apparatus(?: required)?|procedure|result|observation|precautions|theory)\s*:??\s*$"
    )
    matches = list(heading_pattern.finditer(cleaned_text))
    sections = {}

    for index, match in enumerate(matches):
        heading = match.group(1).lower().replace(" required", "")
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(cleaned_text)
        content = cleaned_text[start:end].strip()
        if content:
            sections.setdefault(heading, content)

    return sections, cleaned_text


def render_manual_content(content):
    """Render OCR paragraphs and numbered instructions as readable Markdown."""
    rendered_lines = []
    for line in content.splitlines():
        line = line.strip()
        if re.match(r"^\d+\s*[.)]", line):
            line = re.sub(r"^(\d+)\s*[.)]\s*", r"\1. ", line)
        rendered_lines.append(line)
    st.markdown("\n\n".join(rendered_lines))


def generate_learning_guide(experiment_name, manual_text, known_information=""):
    """Complete missing study sections with Gemini while preserving manual facts."""
    prompt = f"""
You are an expert engineering laboratory educator creating a public learning guide.

Experiment: {experiment_name}

Available laboratory-manual content:
{manual_text}

Known structured information, if available:
{known_information}

Create a well-formatted Markdown study guide with exactly these sections:
## Aim
## Core Concept
## Apparatus and Components
## Important Formulae
## Step-by-Step Procedure
## Expected Observations and Result
## Precautions and Common Mistakes
## Viva Questions and Answers

Use the manual content as the source of truth whenever it contains a fact.
Where the manual is missing or unclear, fill the gap using accurate standard
engineering knowledge. Clearly label general explanations as "Additional learning".
Do not invent measured values, institution-specific wiring, or unsupported claims.
Use simple language, readable bullets, numbered steps, and LaTeX-style formulas.
Return only the study guide in Markdown.
"""
    for model_name in models_to_try:
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=prompt
            )
            if response and response.text:
                return response.text
        except Exception:
            continue
    return None


def get_aec_experiment_chunks(experiment):
    """Return the AEC manual chunks belonging to one catalog experiment."""
    start_page, end_page = experiment["pages"]
    _, chunks = lab_databases["Analog Electronics Circuits"]
    return [
        chunk for chunk in chunks
        if start_page <= int(chunk.get("page", 0)) <= end_page
    ]


def render_aec_repository(search_term):
    """Render AEC experiments as cards with expandable detailed guides."""
    matching_experiments = []
    for experiment in AEC_REPOSITORY_EXPERIMENTS:
        source_text = " ".join(
            chunk.get("text", "")
            for chunk in get_aec_experiment_chunks(experiment)
        )
        searchable_text = f"{experiment['name']} {experiment['summary']} {source_text}".lower()
        if not search_term or search_term in searchable_text:
            matching_experiments.append(experiment)

    if not matching_experiments:
        return 0

    st.markdown("### ⚡ Analog Electronics Circuits Lab")
    st.caption("Browse the AEC manual by experiment. Open a card for the detailed guide.")
    open_experiment = st.session_state.get("aec_repository_open")

    if open_experiment:
        experiment = next(
            (item for item in AEC_REPOSITORY_EXPERIMENTS if item["number"] == open_experiment),
            None
        )
        if experiment:
            if st.button("← Back to AEC experiments", key="aec_repository_back"):
                st.session_state.aec_repository_open = None
                st.rerun()

            chunks = get_aec_experiment_chunks(experiment)
            manual_text = clean_manual_text("\n".join(chunk.get("text", "") for chunk in chunks))
            sections, _ = split_manual_sections(manual_text)
            st.markdown(f"## Experiment {experiment['number']}: {experiment['name']}")
            st.caption(f"Manual pages {experiment['pages'][0]}-{experiment['pages'][1]} · {len(chunks)} indexed chunks")
            st.markdown(experiment["summary"])

            guide_key = f"aec_repository_guide_{experiment['number']}"
            if guide_key not in st.session_state:
                if st.button("Generate detailed guide with Gemini", key=f"generate_{guide_key}"):
                    with st.spinner("Gemini is preparing the detailed AEC guide..."):
                        st.session_state[guide_key] = generate_learning_guide(
                            experiment["name"],
                            manual_text
                        )
                    st.rerun()
            elif st.session_state[guide_key]:
                st.markdown(st.session_state[guide_key])

            display_sections = [
                ("Aim / Objective", ("aim", "objective")),
                ("Apparatus Required", ("apparatus",)),
                ("Procedure", ("procedure",)),
                ("Result and Observations", ("result", "observation")),
                ("Theory and Precautions", ("theory", "precautions"))
            ]
            for title, section_names in display_sections:
                section_content = next(
                    (sections[name] for name in section_names if name in sections),
                    None
                )
                if section_content:
                    st.markdown(f"### {title}")
                    render_manual_content(section_content)
            return 1

    for row_start in range(0, len(matching_experiments), 3):
        columns = st.columns(3)
        for column, experiment in zip(columns, matching_experiments[row_start:row_start + 3]):
            with column:
                st.markdown(
                    f'<article class="feature-card clean dashboard-card">'
                    f'<i>⚡</i><h3>Experiment {experiment["number"]}: {experiment["name"]}</h3>'
                    f'<p>{experiment["summary"]}</p>'
                    f'<small>Manual pages {experiment["pages"][0]}-{experiment["pages"][1]}</small>'
                    f'</article>',
                    unsafe_allow_html=True
                )
                if st.button(
                    "Open detailed guide",
                    key=f"open_aec_repository_{experiment['number']}",
                    use_container_width=True
                ):
                    st.session_state.aec_repository_open = experiment["number"]
                    st.rerun()
    return len(matching_experiments)


def render_learning_repository():
    """Show Communication Systems, AEC, and electronics study repositories."""
    st.markdown("## Learning Repository")
    st.caption(
        "Browse experiment cards and open any experiment for a detailed, manual-grounded guide."
    )

    search_term = st.text_input(
        "Search experiments",
        placeholder="Try: modulation, sampling, apparatus, procedure...",
        key="learning_repository_search"
    ).strip().lower()
    matching_experiments = []

    for experiment_number, experiment in communication_experiments.items():
        procedure = procedure_database.get(experiment_number, {})
        source_text = procedure.get("text", "")
        searchable_text = f"{experiment['name']} {source_text}".lower()
        if not search_term or search_term in searchable_text:
            matching_experiments.append((experiment_number, experiment, procedure))

    electronics_experiments = []
    for experiment_key, experiment_data in knowledge_base.items():
        searchable_text = f"{experiment_key} {json.dumps(experiment_data)}".lower()
        if not search_term or search_term in searchable_text:
            electronics_experiments.append((experiment_key, experiment_data))

    aec_matches = [
        experiment for experiment in AEC_REPOSITORY_EXPERIMENTS
        if not search_term or search_term in experiment["name"].lower() or search_term in experiment["summary"].lower()
    ]
    total_experiments = len(communication_experiments) + len(knowledge_base) + len(AEC_REPOSITORY_EXPERIMENTS)
    st.markdown(
        f"**{len(matching_experiments) + len(electronics_experiments) + len(aec_matches)} of "
        f"{total_experiments} experiments** available"
    )

    if not matching_experiments and not electronics_experiments and not aec_matches:
        st.info("No experiments match your search.")
        return

    for experiment_number, experiment, procedure in matching_experiments:
        sections, cleaned_text = split_manual_sections(procedure.get("text", ""))
        page = procedure.get("procedure_page", "Not specified")
        source = procedure.get("source", "Laboratory manual")

        with st.expander(
            f"Experiment {experiment_number}: {experiment['name']}",
            expanded=len(matching_experiments) == 1
        ):
            guide_key = f"learning_guide_{experiment_number}"
            st.markdown(f"**Source:** {source}  \n**Manual page:** {page}")
            if guide_key not in st.session_state:
                if st.button(
                    "Complete this guide with Gemini",
                    key=f"generate_{guide_key}"
                ):
                    with st.spinner("Gemini is completing the learning guide..."):
                        st.session_state[guide_key] = generate_learning_guide(
                            experiment["name"],
                            cleaned_text
                        )
                    st.rerun()
            else:
                if st.session_state[guide_key]:
                    st.markdown(st.session_state[guide_key])
                else:
                    st.error("Gemini could not complete this guide. Try again later.")
            display_sections = [
                ("Aim / Objective", ("aim", "objective")),
                ("Apparatus Required", ("apparatus",)),
                ("Procedure", ("procedure",)),
                ("Result and Observations", ("result", "observation")),
                ("Theory and Precautions", ("theory", "precautions"))
            ]
            rendered_section = False
            for title, section_names in display_sections:
                section_content = next(
                    (sections[name] for name in section_names if name in sections),
                    None
                )
                if section_content:
                    rendered_section = True
                    st.markdown(f"### {title}")
                    render_manual_content(section_content)

            if not rendered_section:
                st.markdown("### Manual Content")
                render_manual_content(cleaned_text)

            with st.expander("View complete manual transcription"):
                st.text(cleaned_text)

    if electronics_experiments:
        st.markdown("### Electronics Experiment Guides")
        for experiment_key, experiment in electronics_experiments:
            with st.expander(experiment.get("name", experiment_key.replace("_", " ").title())):
                guide_key = f"learning_guide_electronics_{experiment_key}"
                known_information = json.dumps(experiment, indent=2)
                if guide_key not in st.session_state:
                    if st.button(
                        "Complete this guide with Gemini",
                        key=f"generate_{guide_key}"
                    ):
                        with st.spinner("Gemini is completing the learning guide..."):
                            st.session_state[guide_key] = generate_learning_guide(
                                experiment.get("name", experiment_key),
                                "",
                                known_information
                            )
                        st.rerun()
                elif st.session_state[guide_key]:
                    st.markdown(st.session_state[guide_key])
                st.markdown(f"**Formula:** `{experiment.get('formula', 'Not specified')}`")
                st.markdown("### Expected Result")
                st.write(experiment.get("expected_result", "Not specified"))
                st.markdown("### Common Faults")
                for fault in experiment.get("common_faults", []):
                    st.markdown(f"- {fault}")

    render_aec_repository(search_term)


REFERENCE_CIRCUIT_DIR = "reference_circuits"


def preprocess_circuit_image(image_bytes):
    """Create an OpenCV edge view to help identify wiring and component boundaries."""
    image_array = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    if image is None:
        return None

    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(grayscale, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    success, encoded = cv2.imencode(".png", edges)
    return encoded.tobytes() if success else None


def find_reference_circuit(experiment_key):
    """Find the first reference image stored for the selected electronics experiment."""
    if not os.path.isdir(REFERENCE_CIRCUIT_DIR):
        return None

    supported_extensions = {".png", ".jpg", ".jpeg", ".webp"}
    matching_files = []
    for root, _, filenames in os.walk(REFERENCE_CIRCUIT_DIR):
        for filename in filenames:
            extension = os.path.splitext(filename)[1].lower()
            normalized_name = os.path.splitext(filename)[0].lower().replace(" ", "_")
            relative_root = os.path.relpath(root, REFERENCE_CIRCUIT_DIR).lower()
            if extension in supported_extensions and (
                experiment_key in normalized_name or experiment_key in relative_root
            ):
                matching_files.append(os.path.join(root, filename))

    return sorted(matching_files)[0] if matching_files else None


def render_diagnostic_inputs(experiment_key):
    """Render and collect the measurements needed for the selected experiment."""
    input_definitions = EXPERIMENT_DIAGNOSTIC_INPUTS.get(experiment_key, [])
    values = {}
    st.markdown("### Measurements for this experiment")
    st.caption("Enter the values below so LabAI can compare your circuit with the expected behaviour.")
    columns = st.columns(2)
    for index, input_definition in enumerate(input_definitions):
        with columns[index % 2]:
            values[input_definition["key"]] = st.number_input(
                input_definition["label"],
                min_value=0.0,
                value=None,
                placeholder="Enter a measured value",
                help=input_definition["help"],
                key=f"diagnostic_{experiment_key}_{input_definition['key']}"
            )
    return values


# ============================================================
# GEMINI RAG ANSWER
# ============================================================

def generate_grounded_answer(question, retrieved_chunks, lab_name, sections=None):
    """Generate an answer using only the supplied retrieved chunk text."""
    context = "\n\n".join(
        f"Experiment: {chunk['experiment']}\n"
        f"Manual Page: {chunk['page']}\n\n{chunk['text']}"
        for chunk in retrieved_chunks
    )
    section_instruction = ""
    if sections:
        section_instruction = (
            "Return exactly these Markdown sections, in this order:\n"
            + "\n".join(f"## {section}" for section in sections)
        )

    prompt = f"""
You are an AI assistant for the {lab_name} laboratory.

Student question:
{question}

Retrieved laboratory-manual chunks (the only source of truth):
{context}

Answer only from the retrieved chunks. If the chunks do not contain the
answer, say that the laboratory manual does not provide enough information.
Do not invent facts, formulas, values, procedures, or observations.
Do not mention or display raw chunk text, JSON, or retrieval metadata.
{section_instruction}
"""

    for model_name in models_to_try:
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=prompt
            )
            if response and response.text:
                return response.text
        except Exception:
            continue
    return None

def answer_from_lab_manual(question):

    results = []
    experiment_number = detect_communication_experiment(
        question
    )

    # --------------------------------------------------------
    # PROCEDURE QUESTION
    # --------------------------------------------------------

    is_procedure = any(
        word in question.lower()
        for word in [
            "procedure",
            "steps",
            "how to perform",
            "how to do"
        ]
    )

    if (
        is_procedure
        and experiment_number is not None
    ):

        procedure_data = get_procedure(
            experiment_number
        )

        if procedure_data is not None:

            context = json.dumps(
                procedure_data,
                indent=2
            )

        else:

            results = search_lab_manual(
                question,
                k=5
            )

            context = "\n\n".join(
                [
                    result["text"]
                    for result in results
                ]
            )

    else:

        results = search_lab_manual(
            question,
            k=5
        )

        context = "\n\n".join(
            [
                f"""
Experiment: {result['experiment']}
Manual Page: {result['page']}

{result['text']}
"""
                for result in results
            ]
        )

    if results:
        return generate_grounded_answer(
            question,
            results,
            "Communication Systems"
        )

    # Procedure answers retain the existing structured procedure lookup.
    # --------------------------------------------------------
    # GEMINI PROMPT
    # --------------------------------------------------------

    prompt = f"""
You are an AI assistant for a Communication Systems laboratory.

The laboratory manual is the SOURCE OF TRUTH.

Student question:
{question}

Relevant information retrieved from the laboratory manual:

{context}

IMPORTANT RULES:

1. Answer using the retrieved laboratory manual information.
2. Do not invent information that is not supported by the manual.
3. Do not change numerical values from the manual.
4. Do not change formulas from the manual.
5. Preserve procedure order when explaining procedures.
6. If OCR text is unclear, say that the manual text is unclear.
7. Do not guess missing values.
8. Keep the explanation simple and suitable for a laboratory student.
9. Mention the experiment and manual page when useful.

Answer the student's question clearly.
"""

    response = None

    for model_name in models_to_try:

        try:

            response = client.models.generate_content(
                model=model_name,
                contents=prompt
            )

            break

        except Exception:
            continue

    if response:

        return response.text

    return None

def render_structured_procedure(procedure):
    """Render the existing procedure data without exposing its JSON representation."""
    if isinstance(procedure, dict):
        for title, value in procedure.items():
            label = str(title).replace("_", " ").title()
            with st.expander(label, expanded=label.lower() in ("objective", "procedure")):
                render_structured_procedure(value)
    elif isinstance(procedure, list):
        for index, item in enumerate(procedure, 1):
            if isinstance(item, (dict, list)):
                st.markdown(f"**Step {index}**")
                render_structured_procedure(item)
            else:
                st.markdown(f"{index}. {item}")
    else:
        st.markdown(str(procedure))


def render_chat_response(answer, sources, show_context=True):
    st.markdown('<div class="chat-assistant"><span>LABAI</span></div>', unsafe_allow_html=True)
    st.markdown(answer)
    if sources:
        with st.expander("Retrieved context & sources"):
            for source in sources:
                st.markdown(f"**{source['experiment']}** · Manual page {source['page']} · relevance {source['score']:.2f}")
                if show_context:
                    st.caption(source["text"])


def save_technician_record(record_type, payload):
    try:
        technician_records_collection.insert_one({
            "record_type": record_type,
            "technician_id": st.session_state.user_id,
            "created_at": datetime.now(timezone.utc),
            **payload
        })
        return True
    except Exception as error:
        st.error(f"Could not save technician record: {error}")
        return False


def render_technician_console():
    st.markdown("## Technician Console")
    st.caption("Verify circuits, maintain experiment references, and capture viva notes without exposing student identities.")
    verify_tab, maintenance_tab, viva_tab, references_tab = st.tabs([
        "Circuit verification", "Maintenance log", "Viva notes", "Reference circuits"
    ])

    with verify_tab:
        experiment = st.selectbox(
            "Experiment to verify",
            list(knowledge_base.keys()),
            format_func=lambda key: knowledge_base[key]["name"],
            key="technician_verify_experiment"
        )
        verification_image = st.file_uploader(
            "Upload circuit for verification",
            type=["jpg", "jpeg", "png"],
            key="technician_verify_image"
        )
        verification_status = st.selectbox(
            "Verification result",
            ["Verified", "Needs correction", "Unable to verify"],
            key="technician_verification_status"
        )
        verification_notes = st.text_area("Technician notes", key="technician_verification_notes")
        if st.button("Save verification", key="technician_save_verification"):
            if not verification_image:
                st.warning("Upload a circuit image before saving verification.")
            else:
                saved = save_technician_record("circuit_verification", {
                    "experiment": experiment,
                    "status": verification_status,
                    "notes": verification_notes,
                    "has_image": True
                })
                if saved:
                    st.success("Circuit verification saved.")

    with maintenance_tab:
        maintenance_experiment = st.selectbox(
            "Related experiment",
            list(knowledge_base.keys()),
            format_func=lambda key: knowledge_base[key]["name"],
            key="technician_maintenance_experiment"
        )
        component = st.text_input("Component or equipment", key="technician_component")
        maintenance_status = st.selectbox(
            "Status",
            ["Needs repair", "Under repair", "Available"],
            key="technician_maintenance_status"
        )
        maintenance_notes = st.text_area("Maintenance details", key="technician_maintenance_notes")
        if st.button("Save maintenance log", key="technician_save_maintenance"):
            if not component or not maintenance_notes:
                st.warning("Enter the component and maintenance details.")
            else:
                saved = save_technician_record("maintenance", {
                    "experiment": maintenance_experiment,
                    "component": component,
                    "status": maintenance_status,
                    "notes": maintenance_notes
                })
                if saved:
                    st.success("Maintenance record saved.")

    with viva_tab:
        viva_experiment = st.selectbox(
            "Experiment for viva notes",
            list(knowledge_base.keys()),
            format_func=lambda key: knowledge_base[key]["name"],
            key="technician_viva_experiment"
        )
        viva_topic = st.text_input("Viva topic", key="technician_viva_topic")
        viva_notes = st.text_area("Question, expected answer, or teaching note", key="technician_viva_notes")
        if st.button("Save viva note", key="technician_save_viva"):
            if not viva_topic or not viva_notes:
                st.warning("Enter a viva topic and note.")
            else:
                saved = save_technician_record("viva_note", {
                    "experiment": viva_experiment,
                    "topic": viva_topic,
                    "notes": viva_notes
                })
                if saved:
                    st.success("Viva note saved.")

    with references_tab:
        reference_experiment = st.selectbox(
            "Reference experiment",
            list(knowledge_base.keys()),
            format_func=lambda key: knowledge_base[key]["name"],
            key="technician_reference_experiment"
        )
        reference_upload = st.file_uploader(
            "Upload the correct circuit image",
            type=["jpg", "jpeg", "png", "webp"],
            key="technician_reference_upload"
        )
        if st.button("Save reference image", key="technician_save_reference"):
            if not reference_upload:
                st.warning("Choose a reference image first.")
            else:
                reference_directory = os.path.join(REFERENCE_CIRCUIT_DIR, reference_experiment)
                os.makedirs(reference_directory, exist_ok=True)
                reference_path = os.path.join(reference_directory, reference_upload.name)
                with open(reference_path, "wb") as reference_file:
                    reference_file.write(reference_upload.getvalue())
                save_technician_record("reference_image", {
                    "experiment": reference_experiment,
                    "filename": reference_upload.name
                })
                st.success("Reference image saved for this experiment.")


def render_general_ask_ai():
    st.markdown("## Ask LabAI")
    st.caption("Ask about any Communication Systems experiment. LabAI searches the complete laboratory manual.")
    question = st.chat_input(
        "Ask about any experiment, theory, formula, or procedure...",
        key="overview_ask_ai_chat"
    )
    if question:
        st.chat_message("user").markdown(question)
        with st.spinner("Searching the complete laboratory manual..."):
            answer = answer_from_lab_manual(question)
            sources = search_lab_manual(question, k=5)
        if answer:
            with st.chat_message("assistant"):
                render_chat_response(answer, sources)
        else:
            st.error("Gemini is currently unavailable. Please try again later.")


def render_aec_lab():
    """Render the AEC RAG workspace using the shared retrieval pipeline."""
    st.header("⚡ Analog Electronics Circuits Laboratory")
    st.caption("Ask questions grounded in the AEC laboratory manual.")

    aec_index, aec_chunks = lab_databases["Analog Electronics Circuits"]
    stats = st.columns(3)
    with stats[0]:
        st.metric("AEC Chunks Loaded", len(aec_chunks))
    with stats[1]:
        st.metric("AEC FAISS Status", "Loaded" if aec_index.ntotal else "Empty")
    with stats[2]:
        st.metric("AEC PDF Status", "Available" if os.path.isfile(AEC_PDF_PATH) else "Unavailable")

    aec_question = st.text_input(
        "Ask about Analog Electronics Circuits",
        placeholder=(
            "Try: Explain R-2R DAC, Class B Push Pull Amplifier, "
            "regulated power supply, or crossover distortion"
        ),
        key="aec_question"
    )
    if st.button("Ask AEC LabAI", key="aec_ask_button"):
        if not aec_question.strip():
            st.warning("Please enter a question.")
        else:
            with st.spinner("Searching the AEC laboratory manual..."):
                sources = retrieve_chunks(
                    aec_question,
                    "Analog Electronics Circuits",
                    k=5
                )
                answer = generate_grounded_answer(
                    aec_question,
                    sources,
                    "Analog Electronics Circuits",
                    sections=[
                        "Overview",
                        "Key Concepts",
                        "Procedure",
                        "Important Formula",
                        "Observations",
                        "Viva Questions",
                        "Common Mistakes"
                    ]
                )
            if answer:
                with st.chat_message("assistant"):
                    render_chat_response(answer, sources, show_context=False)
            else:
                st.error("Gemini is currently unavailable. Please try again later.")


def open_ask_ai_workspace():
    st.session_state.workspace_view = "Ask AI"


def render_dashboard():
    with st.sidebar:
        st.markdown('<div class="nav-wordmark">✦ LabAI</div>', unsafe_allow_html=True)
        st.caption(f"Signed in as {st.session_state.user_name}")
        st.markdown(f'<div class="role-badge">{str(st.session_state.user_role).replace("_", " ").title()}</div>', unsafe_allow_html=True)
        st.divider()
        selected_lab = st.selectbox(
            "Select Laboratory",
            [
                "Communication Systems",
                "Analog Electronics Circuits"
            ],
            key="selected_lab"
        )
        workspace_options = [
            "Overview",
            "Learning Repository",
            "Ask AI",
            "Communication Systems Lab",
            "Analog Electronics Circuits Lab",
            "Circuit Diagnosis"
        ]
        if st.session_state.user_role == "technician":
            workspace_options.append("Technician Console")
        view = st.radio("Workspace", workspace_options, key="workspace_view", label_visibility="collapsed")
        st.divider()
        if st.button("Logout", key="clean_logout", use_container_width=True):
            st.session_state.logged_in = False
            st.session_state.user_name = st.session_state.user_role = st.session_state.user_id = None
            st.rerun()

    st.markdown(f'''<div class="dashboard-hero"><div><span>YOUR LAB WORKSPACE</span><h1>Welcome back, {st.session_state.user_name}</h1><p>Continue learning with context from your laboratory manual.</p></div><div class="role-badge">{str(st.session_state.user_role).replace("_", " ").title()}</div></div>''', unsafe_allow_html=True)
    if view == "Overview":
        st.markdown("### Your workspace")
        cards = [("🔬", "Communication Systems Lab", "Explore procedures and ask manual-grounded questions."), ("⚡", "Analog Electronics Circuits", "Explore AEC experiments with manual-grounded answers."), ("🤖", "Ask AI", "Get clear answers with source context."), ("🎓", "Viva Preparation", "Use the lab manual to rehearse concepts."), ("🛠", "Fault Diagnosis", "Inspect electronics experiments with AI."), ("📚", "Research Hub", "Search knowledge across the lab repository.")]
        for row in (cards[:3], cards[3:]):
            cols = st.columns(3)
            for col, (icon, title, text) in zip(cols, row):
                with col:
                    st.markdown(f'<article class="feature-card clean dashboard-card"><i>{icon}</i><h3>{title}</h3><p>{text}</p></article>', unsafe_allow_html=True)
                    if title == "Ask AI":
                        st.button(
                            "Open Ask AI",
                            key="overview_ask_ai_button",
                            use_container_width=True,
                            on_click=open_ask_ai_workspace
                        )
        st.info("Choose a workspace from the sidebar to begin.")
    elif view == "Learning Repository":
        render_learning_repository()
    elif view == "Ask AI":
        render_general_ask_ai()
    elif view == "Communication Systems Lab":
        render_communication_lab()
    elif view == "Analog Electronics Circuits Lab":
        if selected_lab == "Analog Electronics Circuits":
            render_aec_lab()
        else:
            st.info("Select Analog Electronics Circuits above to open this laboratory.")
    elif view == "Technician Console":
        render_technician_console()
    else:
        render_circuit_diagnosis()


def render_communication_lab():
    st.markdown("## Communication Systems Laboratory")
    st.caption("Experiment procedures and answers remain grounded in the available laboratory manual.")
    experiment_number = st.selectbox("Select an experiment", list(communication_experiments.keys()), format_func=lambda x: f"Experiment {x} — {communication_experiments[x]['name']}")
    experiment = communication_experiments[experiment_number]
    procedure = get_procedure(experiment_number)
    left, right = st.columns([1.2, 1])
    with left:
        st.markdown(f'<div class="surface-card"><span>EXPERIMENT {experiment_number}</span><h2>{experiment["name"]}</h2><p>Use the sections below to work through the experiment in a structured format.</p></div>', unsafe_allow_html=True)
    with right:
        st.markdown('<div class="surface-card"><span>KNOWLEDGE BASE</span><h3>Manual-grounded retrieval</h3><p>FAISS searches the loaded manual before Gemini responds.</p></div>', unsafe_allow_html=True)
    st.markdown("### Experiment guide")
    if procedure:
        render_structured_procedure(procedure)
    else:
        st.warning("Procedure information not found.")
    st.markdown("### Ask LabAI")
    question = st.chat_input("Ask about theory, apparatus, procedure, observations, or results…", key="communication_chat")
    if question:
        st.chat_message("user").markdown(question)
        with st.spinner("Searching the laboratory manual…"):
            answer = answer_from_lab_manual(question)
            sources = search_lab_manual(question, k=3)
        if answer:
            with st.chat_message("assistant"):
                render_chat_response(answer, sources)
        else:
            st.error("Gemini is currently unavailable. Please try again later.")
    with st.expander("Repository status"):
        st.write(f"Manual chunks loaded: {len(rag_chunks)}")
        st.write(f"FAISS vectors loaded: {faiss_index.ntotal}")
        st.write("Procedure database: 7 experiments")


def render_circuit_diagnosis():
    st.markdown("## Electronics Experiment Diagnosis")
    experiment = st.selectbox("Select your experiment", ["Band Pass Filter", "Voltage Divider", "LED Circuit"], key="clean_electronics_experiment")
    experiment_key = experiment.lower().replace(" ", "_")
    data = knowledge_base[experiment_key]
    reference_circuit = find_reference_circuit(experiment_key)
    info, faults = st.columns(2)
    with info:
        st.markdown(f'<div class="surface-card"><span>EXPECTED BEHAVIOUR</span><h3>{experiment}</h3><p><b>Formula:</b> {data["formula"]}</p><p><b>Expected result:</b> {data["expected_result"]}</p></div>', unsafe_allow_html=True)
        if reference_circuit:
            st.image(reference_circuit, caption=f"Correct {experiment} reference circuit", use_container_width=True)
        else:
            st.info(f"Add a correct reference image to {REFERENCE_CIRCUIT_DIR}/{experiment_key}.png to show it here.")
    with faults:
        st.markdown('<div class="surface-card"><span>COMMON FAULTS</span><h3>What to check</h3></div>', unsafe_allow_html=True)
        for fault in data["common_faults"]:
            st.markdown(f"- {fault}")
    photo, readings = st.columns(2)
    with photo:
        uploaded_image = st.file_uploader("Circuit photo", type=["jpg", "jpeg", "png"], key="clean_circuit_image")
        if uploaded_image:
            uploaded_bytes = uploaded_image.getvalue()
            st.image(uploaded_bytes, caption="Uploaded circuit", use_container_width=True)
            edge_image = preprocess_circuit_image(uploaded_bytes)
            if edge_image:
                st.image(edge_image, caption="OpenCV edge analysis", use_container_width=True)
    with readings:
        diagnostic_inputs = render_diagnostic_inputs(experiment_key)
    if st.button("Diagnose circuit", key="clean_diagnose"):
        if not uploaded_image:
            st.warning("Please upload a circuit photo so LabAI can analyze it.")
        else:
            training_information = json.dumps(fault_dataset.get(experiment_key, []), indent=2)
            uploaded_bytes = uploaded_image.getvalue()
            edge_image = preprocess_circuit_image(uploaded_bytes)
            image_part = types.Part.from_bytes(data=uploaded_bytes, mime_type=uploaded_image.type)
            prompt = f'''You are an electronics laboratory diagnostic assistant.

Your task is to diagnose faults in a student's
electronics experiment.

EXPERIMENT:
{experiment}

EXPERIMENT FORMULA:
{data["formula"]}

EXPECTED RESULT:
{data["expected_result"]}

COMMON FAULTS:
{", ".join(data["common_faults"])}

STUDENT MEASUREMENTS:

{json.dumps(diagnostic_inputs, indent=2)}

REFERENCE FAULT CASES:

{training_information}

IMAGE INPUTS:
The first image is the student's uploaded circuit.
The second image, when present, is the OpenCV edge-analysis view.
The final image, when present, is the correct reference circuit for this experiment.

IMPORTANT INSTRUCTIONS:

1. Analyze the student's actual circuit image.
2. Analyze the student's measurements.
3. Use the reference fault cases as diagnostic knowledge.
4. Do not automatically select a reference fault just because it exists.
5. Compare the student's symptoms with possible faults.
6. If the photograph is unclear, clearly say that the wiring cannot be confirmed.
7. Never claim that a connection is definitely wrong if the image does not clearly show it.
8. Give the most likely fault first.
9. Give alternative possible faults when appropriate.
10. Explain what the student should physically check.
11. Use the OpenCV edge-analysis image to inspect wire paths and component boundaries.
12. Compare the uploaded circuit with the correct reference image when one is provided.
13. Clearly separate confirmed visual evidence from likely possibilities.
14. Some measurements may be unavailable. Do not invent missing values; base the diagnosis on the image and available measurements.

Give your answer using exactly these sections:

1. MOST LIKELY FAULT

2. WHY THIS MAY BE HAPPENING

3. EVIDENCE FROM THE CIRCUIT

4. WHAT TO CHECK

5. CORRECTIVE ACTION

6. EXPECTED RESULT

7. CONFIDENCE

Keep the explanation simple for an electronics laboratory student.'''
            response = None
            with st.spinner("AI is analyzing your circuit…"):
                for model_name in models_to_try:
                    try:
                        image_parts = [prompt, image_part]
                        if edge_image:
                            image_parts.append(types.Part.from_bytes(data=edge_image, mime_type="image/png"))
                        if reference_circuit:
                            with open(reference_circuit, "rb") as reference_file:
                                reference_bytes = reference_file.read()
                            reference_type = "image/png" if reference_circuit.lower().endswith(".png") else "image/jpeg"
                            image_parts.append(types.Part.from_bytes(data=reference_bytes, mime_type=reference_type))
                        response = client.models.generate_content(model=model_name, contents=image_parts)
                        break
                    except Exception:
                        continue
            if response:
                with st.chat_message("assistant"):
                    render_chat_response(response.text, [])
            else:
                st.error("Gemini is currently unavailable. Please try again later.")
    st.markdown("### Ask about this experiment")
    question = st.chat_input("Ask a question about this circuit…", key="electronics_chat")
    if question:
        st.chat_message("user").markdown(question)
        training_information = json.dumps(fault_dataset.get(experiment_key, []), indent=2)
        prompt = f'''You are an electronics laboratory teaching assistant.

The student is studying:

Experiment:
{experiment}

Formula:
{data["formula"]}

Expected result:
{data["expected_result"]}

Common faults:
{", ".join(data["common_faults"])}

Reference fault cases:
{training_information}

Student's question:
{question}

Use the experiment information and reference fault cases
to answer the student's question.

Do not invent measurements or circuit connections.

Answer clearly and simply.'''
        response = None
        with st.spinner("LabAI is thinking…"):
            for model_name in models_to_try:
                try:
                    response = client.models.generate_content(model=model_name, contents=prompt)
                    break
                except Exception:
                    continue
        if response:
            with st.chat_message("assistant"):
                render_chat_response(response.text, [])
        else:
            st.error("Gemini is currently unavailable. Please try again later.")


render_dashboard()
st.stop()


with st.sidebar:
    st.markdown("""
    <div style="padding: 10px 0 20px 0; border-bottom: 1px solid rgba(255,255,255,0.1); margin-bottom: 15px;">
        <h2 style="margin: 0; color: #FFFFFF; font-size: 1.4rem; font-weight: 800;">🔬 LabAI Workspace</h2>
        <span style="display: inline-block; background: rgba(99,102,241,0.15); color: #818CF8; border: 1px solid rgba(99,102,241,0.3); font-size: 0.72rem; padding: 2px 8px; border-radius: 6px; font-weight: 600; margin-top: 6px;">
            ● ACTIVE LAB SESSION
        </span>
    </div>
    """, unsafe_allow_html=True)

    st.success(f"👤 {st.session_state.user_name}")
    st.caption(f"Role: {str(st.session_state.user_role).title()}")

    if st.button("🚪 Logout to Landing Page", use_container_width=True, key="sidebar_logout_btn"):
        st.session_state.logged_in = False
        st.session_state.user_name = None
        st.session_state.user_role = None
        st.session_state.user_id = None
        st.session_state.show_auth = False
        st.rerun()
# ============================================================
# MODE SELECTION
# ============================================================

st.sidebar.header("🔧 Assistant Mode")

mode = st.sidebar.radio(
    "Choose what you want to do:",
    [
        "Existing Electronics Experiments",
        "Communication Systems Lab"
    ]
)


# ============================================================
# EXISTING ELECTRONICS EXPERIMENTS
# ============================================================

if mode == "Existing Electronics Experiments":

    st.header("🔌 Electronics Experiment Diagnosis")

    experiment = st.selectbox(
        "Select your experiment:",
        [
            "Band Pass Filter",
            "Voltage Divider",
            "LED Circuit"
        ]
    )

    experiment_key = experiment.lower().replace(
        " ",
        "_"
    )

    data = knowledge_base[experiment_key]


    # --------------------------------------------------------
    # EXPERIMENT INFORMATION
    # --------------------------------------------------------

    st.subheader("Experiment Information")

    st.write(
        "**Formula:**",
        data["formula"]
    )

    st.write(
        "**Expected Result:**",
        data["expected_result"]
    )


    # --------------------------------------------------------
    # COMMON FAULTS
    # --------------------------------------------------------

    st.subheader("Common Faults")

    for fault in data["common_faults"]:

        st.write(
            "•",
            fault
        )


    # --------------------------------------------------------
    # CIRCUIT PHOTO
    # --------------------------------------------------------

    st.subheader("Circuit Photo")

    uploaded_image = st.file_uploader(
        "Upload a photo of your circuit:",
        type=[
            "jpg",
            "jpeg",
            "png"
        ],
        key="existing_circuit_image"
    )

    if uploaded_image:

        st.image(
            uploaded_image,
            caption="Uploaded Circuit",
            width="stretch"
        )


    # --------------------------------------------------------
    # MEASUREMENTS
    # --------------------------------------------------------

    st.subheader("📊 Measurements")

    input_voltage = st.number_input(
        "Input Voltage (V):",
        min_value=0.0,
        step=0.1
    )

    output_voltage = st.number_input(
        "Output Voltage (V):",
        min_value=0.0,
        step=0.1
    )

    frequency = st.number_input(
        "Frequency (Hz):",
        min_value=0.0,
        step=1.0
    )


    # --------------------------------------------------------
    # OBSERVED RESULT
    # --------------------------------------------------------

    st.subheader("Observed Result")

    observed_result = st.text_input(
        "Enter your measured result:",
        placeholder="Example: 15 kHz"
    )


    # --------------------------------------------------------
    # CIRCUIT DIAGNOSIS
    # --------------------------------------------------------

    st.divider()

    st.header("🔍 Circuit Diagnosis")

    if st.button(
        "Diagnose Circuit",
        key="diagnose_existing"
    ):

        if not uploaded_image:

            st.warning(
                "Please upload a circuit photo."
            )

        elif not observed_result:

            st.warning(
                "Please enter your observed result."
            )

        else:

            with st.spinner(
                "AI is analyzing your circuit..."
            ):

                training_cases = fault_dataset.get(
                    experiment_key,
                    []
                )

                training_information = json.dumps(
                    training_cases,
                    indent=2
                )

                image_part = types.Part.from_bytes(
                    data=uploaded_image.getvalue(),
                    mime_type=uploaded_image.type
                )

                prompt = f"""
You are an electronics laboratory diagnostic assistant.

Your task is to diagnose faults in a student's
electronics experiment.

EXPERIMENT:
{experiment}

EXPERIMENT FORMULA:
{data["formula"]}

EXPECTED RESULT:
{data["expected_result"]}

COMMON FAULTS:
{", ".join(data["common_faults"])}

STUDENT MEASUREMENTS:

Input Voltage:
{input_voltage} V

Output Voltage:
{output_voltage} V

Frequency:
{frequency} Hz

Student's observed result:
{observed_result}

REFERENCE FAULT CASES:

{training_information}

IMPORTANT INSTRUCTIONS:

1. Analyze the student's actual circuit image.
2. Analyze the student's measurements.
3. Use the reference fault cases as diagnostic knowledge.
4. Do not automatically select a reference fault just because it exists.
5. Compare the student's symptoms with possible faults.
6. If the photograph is unclear, clearly say that the wiring cannot be confirmed.
7. Never claim that a connection is definitely wrong if the image does not clearly show it.
8. Give the most likely fault first.
9. Give alternative possible faults when appropriate.
10. Explain what the student should physically check.

Give your answer using exactly these sections:

1. MOST LIKELY FAULT

2. WHY THIS MAY BE HAPPENING

3. EVIDENCE FROM THE CIRCUIT

4. WHAT TO CHECK

5. CORRECTIVE ACTION

6. EXPECTED RESULT

7. CONFIDENCE

Keep the explanation simple for an electronics laboratory student.
"""

                response = None

                for model_name in models_to_try:

                    try:

                        response = client.models.generate_content(
                            model=model_name,
                            contents=[
                                prompt,
                                image_part
                            ]
                        )

                        break

                    except Exception:
                        continue


                if response:

                    st.subheader(
                        "🤖 AI Diagnosis"
                    )

                    st.write(
                        response.text
                    )

                else:

                    st.error(
                        "Gemini is currently unavailable. "
                        "Please try again later."
                    )


    # --------------------------------------------------------
    # EXISTING ASK AI
    # --------------------------------------------------------

    st.divider()

    st.header(
        "💬 Ask AI About This Experiment"
    )

    question = st.text_input(
        "Your question:",
        placeholder="Example: Why is my output voltage different?",
        key="existing_question"
    )

    if st.button(
        "Ask AI",
        key="existing_ask_ai"
    ):

        if not question:

            st.warning(
                "Please enter a question."
            )

        else:

            with st.spinner(
                "AI is thinking..."
            ):

                training_cases = fault_dataset.get(
                    experiment_key,
                    []
                )

                training_information = json.dumps(
                    training_cases,
                    indent=2
                )

                chat_prompt = f"""
You are an electronics laboratory teaching assistant.

The student is studying:

Experiment:
{experiment}

Formula:
{data["formula"]}

Expected result:
{data["expected_result"]}

Common faults:
{", ".join(data["common_faults"])}

Reference fault cases:
{training_information}

Student's question:
{question}

Use the experiment information and reference fault cases
to answer the student's question.

Do not invent measurements or circuit connections.

Answer clearly and simply.
"""

                response = None

                for model_name in models_to_try:

                    try:

                        response = client.models.generate_content(
                            model=model_name,
                            contents=chat_prompt
                        )

                        break

                    except Exception:
                        continue


                if response:

                    st.subheader(
                        "🤖 AI Answer"
                    )

                    st.write(
                        response.text
                    )

                else:

                    st.error(
                        "Gemini is currently unavailable. "
                        "Please try again later."
                    )


# ============================================================
# COMMUNICATION SYSTEMS LAB
# ============================================================

else:

    st.header(
        "📡 Communication Systems Laboratory"
    )

    st.write(
        "Ask questions about the 7 Communication Systems "
        "experiments using the laboratory manual."
    )


    # --------------------------------------------------------
    # SELECT EXPERIMENT
    # --------------------------------------------------------

    selected_experiment_number = st.selectbox(

        "Select Communication Systems experiment:",

        options=list(
            communication_experiments.keys()
        ),

        format_func=lambda x:
            f"Experiment {x} — "
            f"{communication_experiments[x]['name']}"
    )


    selected_experiment = communication_experiments[
        selected_experiment_number
    ]


    st.info(
        f"Selected: Experiment "
        f"{selected_experiment_number} — "
        f"{selected_experiment['name']}"
    )


    # --------------------------------------------------------
    # PROCEDURE BUTTON
    # --------------------------------------------------------

    st.subheader(
        "📋 Experiment Procedure"
    )

    if st.button(
        "Show Procedure",
        key="show_procedure"
    ):

        procedure = get_procedure(
            selected_experiment_number
        )

        if procedure:

            st.write(
                procedure
            )

        else:

            st.warning(
                "Procedure information not found."
            )


    # --------------------------------------------------------
    # ASK QUESTION
    # --------------------------------------------------------

    st.subheader(
        "💬 Ask About This Experiment"
    )

    communication_question = st.text_input(

        "Enter your question:",

        placeholder=(
            "Example: What is the procedure for "
            "Frequency Modulation?"
        ),

        key="communication_question"
    )


    if st.button(
        "Ask AI",
        key="communication_ask_ai"
    ):

        if not communication_question:

            st.warning(
                "Please enter a question."
            )

        else:

            with st.spinner(
                "Searching the laboratory manual..."
            ):

                answer = answer_from_lab_manual(
                    communication_question
                )


            if answer:

                st.subheader(
                    "🤖 AI Answer"
                )

                st.write(
                    answer
                )

            else:

                st.error(
                    "Gemini is currently unavailable. "
                    "Please try again later."
                )


    # --------------------------------------------------------
    # RAG DATABASE INFORMATION
    # --------------------------------------------------------

    st.divider()

    st.subheader(
        "📚 RAG Knowledge Base"
    )

    st.write(
        f"Manual chunks loaded: "
        f"{len(rag_chunks)}"
    )

    st.write(
        f"FAISS vectors loaded: "
        f"{faiss_index.ntotal}"
    )

    st.write(
        "Procedure database: 7 experiments"
    )
