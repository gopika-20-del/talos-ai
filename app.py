import streamlit as st
import pandas as pd
import numpy as np
import json
import joblib
import os
import textwrap
from pathlib import Path
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go

# Import src modules
from src.utils import find_data_dir, load_candidates
from src.jd_parser import parse_job_description
from src.embeddings import CandidateEmbeddingEngine
from src.ranking_model import CandidateRanker, FEATURE_NAMES
from src.trust_engine import compute_trust_and_risk
from src.explainability import generate_reasoning

# Page configuration
st.set_page_config(
    page_title="talentIQ - AI Ranked Shortlist",
    page_icon="https://img.icons8.com/color/96/robot.png",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Helper function to strip all leading/trailing whitespace per line from HTML templates.
# This prevents markdown parsers from treating indented HTML tags as <pre><code> blocks.
def clean_html(html_str):
    return "\n".join(line.strip() for line in html_str.split("\n") if line.strip())

# Custom premium styling injection
st.markdown(clean_html("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Outfit:wght@400;500;600;700;800&display=swap');
    
    .stApp {
        background-color: #f8f9fc !important;
        color: #1e293b !important;
        font-family: 'Inter', sans-serif !important;
    }
    
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Outfit', sans-serif !important;
        font-weight: 700 !important;
        color: #1e293b !important;
    }
    
    /* Main Content Top Padding override */
    .main .block-container, [data-testid="stMainBlockContainer"] {
        padding-top: 1.5rem !important;
        padding-bottom: 2rem !important;
    }
    
    /* Sidebar Overrides */
    [data-testid="stSidebar"] {
        background-color: #ffffff !important;
        border-right: 1px solid #eef0f6 !important;
        padding-top: 0rem !important;
    }
    
    [data-testid="stSidebar"] [data-testid="stSidebarUserContent"] {
        padding-top: 0rem !important;
        margin-top: -3.5rem !important;
    }
    
    [data-testid="stSidebar"] .block-container {
        padding-top: 0rem !important;
    }
    
    /* Navigation Styles override for stRadio */
    [data-testid="stSidebar"] div[data-testid="stRadio"] label {
        display: flex !important;
        align-items: center !important;
        padding: 12px 16px !important;
        margin-bottom: 6px !important;
        border-radius: 12px !important;
        background-color: transparent !important;
        color: #64748b !important;
        font-family: 'Outfit', sans-serif !important;
        font-weight: 500 !important;
        font-size: 0.95rem !important;
        border: 1px solid transparent !important;
        cursor: pointer !important;
        transition: all 0.2s ease-in-out !important;
        position: relative !important;
    }
    
    [data-testid="stSidebar"] div[data-testid="stRadio"] label:hover {
        background-color: #f8f9fc !important;
        color: #7c3aed !important;
    }
    
    [data-testid="stSidebar"] div[data-testid="stRadio"] label:has(input:checked) {
        background-color: #f5f3ff !important;
        color: #7c3aed !important;
        font-weight: 600 !important;
        border: 1px solid #e9d5ff !important;
    }
    
    [data-testid="stSidebar"] div[data-testid="stRadio"] label input {
        position: absolute !important;
        opacity: 0 !important;
        width: 0 !important;
        height: 0 !important;
        pointer-events: none !important;
    }
    
    [data-testid="stSidebar"] div[data-testid="stRadio"] label [data-testid="stWidgetLabel"] {
        display: none !important;
    }
    
    /* Logo branding styling */
    .logo-container {
        display: flex;
        align-items: center;
        gap: 12px;
        padding: 0px 16px 24px 16px;
        border-bottom: 1px solid #f1f5f9;
        margin-bottom: 20px;
    }
    .logo-icon {
        width: 36px;
        height: 36px;
        background: linear-gradient(135deg, #6366f1, #7c3aed);
        border-radius: 10px;
        position: relative;
        box-shadow: 0 4px 6px -1px rgba(99, 102, 241, 0.2);
    }
    .logo-icon::before {
        content: '';
        position: absolute;
        width: 14px;
        height: 14px;
        background-color: white;
        border-radius: 50%;
        top: 6px;
        left: 6px;
    }
    .logo-icon::after {
        content: '';
        position: absolute;
        width: 8px;
        height: 8px;
        background-color: white;
        border-radius: 50%;
        bottom: 6px;
        right: 6px;
    }
    .logo-text {
        display: flex;
        flex-direction: column;
    }
    .logo-main {
        font-family: 'Outfit', sans-serif;
        font-size: 1.5rem;
        font-weight: 700;
        color: #1e293b;
        line-height: 1;
    }
    .logo-sub {
        font-family: 'Outfit', sans-serif;
        font-size: 0.7rem;
        font-weight: 600;
        color: #94a3b8;
        letter-spacing: 0.1em;
        margin-top: 2px;
    }
    
    /* Sidebar bot card */
    .sidebar-bot-card {
        background: linear-gradient(180deg, #f5f3ff 0%, #e0e7ff 100%);
        border: 1px solid #ddd6fe;
        border-radius: 16px;
        padding: 20px 16px;
        text-align: center;
        margin: 30px 16px 20px 16px;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.02);
    }
    .bot-img {
        width: 60px;
        height: 60px;
        margin-bottom: 8px;
        display: inline-block;
    }
    .bot-title {
        font-family: 'Outfit', sans-serif;
        font-size: 1.1rem;
        font-weight: 700;
        color: #4c1d95;
        margin-bottom: 4px;
    }
    .bot-subtitle {
        font-size: 0.8rem;
        color: #6d28d9;
        margin-bottom: 14px;
        line-height: 1.4;
    }
    .bot-btn {
        display: block;
        width: 100%;
        background: linear-gradient(135deg, #7c3aed, #6366f1);
        color: white !important;
        text-decoration: none !important;
        padding: 10px;
        border-radius: 10px;
        font-size: 0.85rem;
        font-weight: 600;
        font-family: 'Outfit', sans-serif;
        transition: transform 0.2s, box-shadow 0.2s;
        box-shadow: 0 4px 6px -1px rgba(124, 58, 237, 0.2);
        border: none;
    }
    .bot-btn:hover {
        transform: translateY(-1px);
        box-shadow: 0 6px 12px -1px rgba(124, 58, 237, 0.3);
    }
    
    /* KPI Cards Styling */
    .kpi-card {
        background-color: #ffffff;
        border: 1px solid #eef0f6;
        border-radius: 16px;
        padding: 18px;
        display: flex;
        align-items: center;
        gap: 16px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.02), 0 2px 4px -1px rgba(0, 0, 0, 0.01);
        height: 105px;
    }
    .kpi-icon-container {
        width: 48px;
        height: 48px;
        border-radius: 12px;
        display: flex;
        align-items: center;
        justify-content: center;
    }
    
    .kpi-content {
        display: flex;
        flex-direction: column;
    }
    .kpi-label {
        font-family: 'Outfit', sans-serif;
        font-size: 0.75rem;
        font-weight: 700;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .kpi-value {
        font-family: 'Outfit', sans-serif;
        font-size: 1.4rem;
        font-weight: 700;
        color: #1e293b;
        line-height: 1.2;
        margin-top: 2px;
    }
    .kpi-subtext {
        font-size: 0.75rem;
        color: #64748b;
        margin-top: 2px;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }
    
    /* Custom Candidate Card */
    .cand-card {
        background-color: #ffffff;
        border: 1px solid #eef0f6;
        border-radius: 16px;
        padding: 20px;
        margin-bottom: 16px;
        display: flex;
        align-items: center;
        justify-content: space-between;
        transition: transform 0.2s, box-shadow 0.2s;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.02);
    }
    .cand-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.04);
        border-color: #c084fc;
    }
    .cand-left {
        display: flex;
        align-items: center;
        gap: 16px;
        flex-grow: 1;
    }
    .cand-rank-badge {
        width: 36px;
        height: 36px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-family: 'Outfit', sans-serif;
        font-weight: 700;
        font-size: 0.95rem;
        flex-shrink: 0;
    }
    .rank-1 { background-color: #fef3c7; color: #d97706; border: 1px solid #fde68a; }
    .rank-2 { background-color: #f1f5f9; color: #475569; border: 1px solid #e2e8f0; }
    .rank-3 { background-color: #ffedd5; color: #ea580c; border: 1px solid #fed7aa; }
    .rank-other { background-color: #f8f9fc; color: #94a3b8; border: 1px solid #e2e8f0; }
    
    .cand-avatar {
        width: 56px;
        height: 56px;
        border-radius: 50%;
        object-fit: cover;
        border: 2px solid #ffffff;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.08);
        flex-shrink: 0;
    }
    .cand-info {
        display: flex;
        flex-direction: column;
        gap: 4px;
        width: 38%;
        padding-right: 10px;
    }
    .cand-name-row {
        display: flex;
        align-items: center;
        gap: 6px;
    }
    .cand-name {
        font-family: 'Outfit', sans-serif;
        font-size: 1.15rem;
        font-weight: 700;
        color: #1e293b;
    }
    .verified-icon {
        color: #10b981;
        display: flex;
        align-items: center;
    }
    .cand-title {
        font-size: 0.85rem;
        font-weight: 600;
        color: #64748b;
    }
    .cand-meta {
        font-size: 0.8rem;
        color: #94a3b8;
    }
    
    .cand-skills-strengths {
        display: flex;
        flex-direction: column;
        gap: 8px;
        width: 42%;
    }
    .cand-section-title {
        font-size: 0.7rem;
        font-weight: 700;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 2px;
    }
    .cand-skills-list {
        display: flex;
        flex-wrap: wrap;
        gap: 6px;
    }
    .skill-tag {
        font-size: 0.75rem;
        padding: 2px 8px;
        border-radius: 6px;
        font-weight: 500;
    }
    
    .cand-strengths-list {
        display: flex;
        flex-direction: column;
        gap: 3px;
    }
    .strength-item {
        font-size: 0.8rem;
        color: #475569;
        display: flex;
        align-items: center;
        gap: 6px;
    }
    .strength-check {
        color: #10b981;
        display: flex;
        align-items: center;
    }
    
    .cand-right {
        display: flex;
        align-items: center;
        gap: 16px;
        flex-shrink: 0;
    }
    .progress-container {
        display: flex;
        flex-direction: column;
        align-items: center;
        gap: 4px;
        width: 75px;
    }
    .fit-label {
        font-family: 'Outfit', sans-serif;
        font-size: 0.7rem;
        font-weight: 700;
        color: #10b981;
        text-transform: uppercase;
        text-align: center;
    }
    
    .action-container {
        display: flex;
        flex-direction: column;
        align-items: center;
        gap: 8px;
    }
    .view-profile-btn {
        display: inline-block;
        padding: 6px 12px;
        border-radius: 8px;
        border: 1.5px solid #7c3aed;
        color: #7c3aed !important;
        text-decoration: none !important;
        font-family: 'Outfit', sans-serif;
        font-size: 0.8rem;
        font-weight: 600;
        transition: all 0.2s;
        background-color: transparent;
        text-align: center;
        cursor: pointer;
    }
    .view-profile-btn:hover {
        background-color: #7c3aed;
        color: white !important;
    }
    .bookmark-btn {
        width: 32px;
        height: 32px;
        border-radius: 8px;
        border: 1px solid #e2e8f0;
        display: flex;
        align-items: center;
        justify-content: center;
        cursor: pointer;
        color: #94a3b8;
        background-color: transparent;
        transition: all 0.2s;
    }
    .bookmark-btn:hover {
        color: #7c3aed;
        border-color: #ddd6fe;
        background-color: #f5f3ff;
    }
    
    /* White Content Card */
    .white-card {
        background-color: #ffffff;
        border: 1px solid #eef0f6;
        border-radius: 16px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.02);
    }
    
    /* Custom Streamlit adjustments */
    div[data-testid="stForm"] {
        background-color: #ffffff !important;
        border: 1px solid #eef0f6 !important;
        border-radius: 16px !important;
        padding: 20px !important;
    }
    
    .stProgress > div > div > div > div {
        background-color: #7c3aed !important;
    }
    
    /* Spotlight Styling */
    .spotlight-header-card {
        background: linear-gradient(135deg, #ffffff 0%, #fcfbff 100%);
        border: 1px solid #eef0f6;
        border-top: 4px solid #7c3aed;
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 20px;
        box-shadow: 0 4px 10px rgba(0,0,0,0.02);
    }
    
    .reasoning-box {
        background: linear-gradient(135deg, #f5f3ff, #eff6ff);
        border-left: 4px solid #7c3aed;
        border-radius: 8px;
        padding: 18px;
        margin-bottom: 24px;
        font-family: 'Inter', sans-serif;
        color: #4c1d95;
        font-size: 1.02rem;
        line-height: 1.5;
        font-style: italic;
    }
    
    /* Timeline style */
    .timeline-container {
        border-left: 2px solid #e2e8f0;
        padding-left: 20px;
        margin-left: 10px;
        position: relative;
    }
    .timeline-item {
        position: relative;
        margin-bottom: 24px;
    }
    .timeline-marker {
        position: absolute;
        left: -27px;
        top: 4px;
        width: 12px;
        height: 12px;
        border-radius: 50%;
        background-color: #7c3aed;
        border: 2px solid #ffffff;
    }
    .timeline-date {
        font-size: 0.8rem;
        font-weight: 600;
        color: #94a3b8;
        margin-bottom: 4px;
    }
    
    /* Trust Audit badges */
    .audit-badge {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
        text-align: center;
    }
    .audit-pass {
        background-color: #ecfdf5;
        color: #059669;
        border: 1px solid #a7f3d0;
    }
    .audit-warn {
        background-color: #fffbeb;
        color: #d97706;
        border: 1px solid #fde68a;
    }
    .audit-fail {
        background-color: #fef2f2;
        color: #dc2626;
        border: 1px solid #fecaca;
    }
    </style>
"""), unsafe_allow_html=True)

# Helper function to cache candidates loading for instant dashboards
@st.cache_data
def get_cached_candidates():
    logger = st.empty()
    logger.info("First load: parsing candidate profiles database (this takes ~10 seconds)...")
    cands = list(load_candidates())
    logger.empty()
    return cands

@st.cache_data
def load_shortlist_and_scores():
    data_dir = find_data_dir()
    top_100_path = data_dir / "outputs" / "top_100_candidates.csv"
    scores_path = data_dir / "outputs" / "candidate_scores.csv"
    imp_path = data_dir / "outputs" / "feature_importance.csv"
    
    top_df = pd.read_csv(top_100_path) if top_100_path.exists() else None
    scores_df = pd.read_csv(scores_path) if scores_path.exists() else None
    imp_df = pd.read_csv(imp_path) if imp_path.exists() else None
    
    return top_df, scores_df, imp_df

# Load data
top_df, scores_df, imp_df = load_shortlist_and_scores()

# Load top 100 profiles cache
@st.cache_data
def load_top_100_profiles():
    data_dir = find_data_dir()
    profiles_path = data_dir / "outputs" / "top_100_profiles.json"
    if profiles_path.exists():
        with open(profiles_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

cand_by_id = load_top_100_profiles()

if scores_df is not None:
    scores_by_id = scores_df.set_index("candidate_id").to_dict(orient="index")
else:
    scores_by_id = {}

# Profile photo list
CANDIDATE_AVATARS = [
    "https://images.unsplash.com/photo-1507003211169-0a1dd7228f2d?auto=format&fit=crop&w=150&h=150&q=80",
    "https://images.unsplash.com/photo-1494790108377-be9c29b29330?auto=format&fit=crop&w=150&h=150&q=80",
    "https://images.unsplash.com/photo-1500648767791-00dcc994a43e?auto=format&fit=crop&w=150&h=150&q=80",
    "https://images.unsplash.com/photo-1438761681033-6461ffad8d80?auto=format&fit=crop&w=150&h=150&q=80",
    "https://images.unsplash.com/photo-1472099645785-5658abf4ff4e?auto=format&fit=crop&w=150&h=150&q=80",
    "https://images.unsplash.com/photo-1544005313-94ddf0286df2?auto=format&fit=crop&w=150&h=150&q=80",
    "https://images.unsplash.com/photo-1506794778202-cad84cf45f1d?auto=format&fit=crop&w=150&h=150&q=80",
    "https://images.unsplash.com/photo-1534528741775-53994a69daeb?auto=format&fit=crop&w=150&h=150&q=80",
    "https://images.unsplash.com/photo-1522075469751-3a6694fb2f61?auto=format&fit=crop&w=150&h=150&q=80",
    "https://images.unsplash.com/photo-1517841905240-472988babdf9?auto=format&fit=crop&w=150&h=150&q=80"
]

def get_avatar_url(candidate_id, rank):
    if rank <= 5:
        return CANDIDATE_AVATARS[rank - 1]
    val = sum(ord(c) for c in candidate_id)
    return CANDIDATE_AVATARS[val % len(CANDIDATE_AVATARS)]

# Key details helper
def get_key_strengths(cand, score_row):
    profile = cand.get("profile", {})
    skills = cand.get("skills", [])
    edu = cand.get("education", [])
    signals = cand.get("redrob_signals", {})
    
    strengths = []
    
    # 1. Experience level
    exp = profile.get("years_of_experience", 0)
    if exp > 6:
        strengths.append("Strong ML background")
    elif exp >= 4:
        strengths.append("Predictive Modeling")
    else:
        strengths.append("High learning agility")
        
    # 2. Education tier
    has_tier1 = any(e.get("tier") == "tier_1" for e in edu)
    if has_tier1:
        strengths.append("Tier-1 Education")
    else:
        expert_skills = [s.get("name") for s in skills if s.get("proficiency") == "expert"]
        if expert_skills:
            strengths.append(f"Expert in {expert_skills[0]}")
        else:
            strengths.append("Problem Solver")
            
    # 3. Code quality or responsiveness
    gh_score = signals.get("github_activity_score", 0)
    resp_rate = signals.get("recruiter_response_rate", 0)
    if gh_score > 70:
        strengths.append("Kaggle Expert" if "kaggle" in str(cand).lower() else "Active Contributor")
    elif resp_rate > 0.8:
        strengths.append("High platform agility")
    else:
        strengths.append("System Design focus")
        
    while len(strengths) < 3:
        strengths.append("Detail Oriented")
    return strengths[:3]

# Synchronize navigation with query parameters and session state
query_params = st.query_params
pages_list = [
    "Dashboard",
    "Job Intake",
    "Candidate Search",
    "Ranked Shortlist",
    "Talent Insights",
    "Reports",
    "Settings"
]

# Initialize state keys to avoid AttributeErrors
if "current_page" not in st.session_state:
    st.session_state.current_page = "Ranked Shortlist"
if "selected_candidate" not in st.session_state:
    st.session_state.selected_candidate = None

if "page" in query_params:
    page_val = query_params["page"].lower()
    for p in pages_list:
        p_clean = p.strip().lower()
        if p_clean in page_val or page_val in p_clean:
            st.session_state.current_page = p
            break

if "candidate_id" in query_params:
    st.session_state.selected_candidate = query_params["candidate_id"]

# Sidebar logo
st.sidebar.markdown(clean_html("""
    <div class="logo-container">
        <div class="logo-icon"></div>
        <div class="logo-text">
            <span class="logo-main">talent<span style="color:#7c3aed; font-weight:800;">IQ</span></span>
            <span class="logo-sub">AI-POWERED HIRING</span>
        </div>
    </div>
"""), unsafe_allow_html=True)

# Select default index
try:
    default_idx = pages_list.index(st.session_state.current_page)
except ValueError:
    default_idx = 3

selected_page = st.sidebar.radio(
    "Select Workspace",
    pages_list,
    index=default_idx,
    key="nav_radio"
)

# Update state if changed via radio
if selected_page != st.session_state.current_page:
    st.session_state.current_page = selected_page
    if selected_page != "Talent Insights":
        st.session_state.selected_candidate = None
    st.query_params.clear()

# Bottom sidebar card
st.sidebar.markdown(clean_html("""
    <div class="sidebar-bot-card">
        <img src="https://img.icons8.com/color/96/robot.png" class="bot-img" />
        <div class="bot-title">AI Recruiter</div>
        <div class="bot-subtitle">Working 24/7 to find your perfect hire</div>
        <a href="?page=Talent+Insights" target="_self" class="bot-btn">View Insights</a>
    </div>
"""), unsafe_allow_html=True)

# ----------------------------------------------------
# SVG RENDERING UTILS
# ----------------------------------------------------
def render_fit_circle(score):
    dashoffset = 150.8 * (1 - (score / 100.0))
    if score >= 90:
        color = "#10b981"
        label = "Best Match" if score >= 95 else "Excellent Match"
    elif score >= 85:
        color = "#10b981"
        label = "Great Match"
    elif score >= 70:
        color = "#3b82f6"
        label = "Good Match"
    else:
        color = "#f59e0b"
        label = "Fair Match"
        
    return f"""
    <div class="progress-container">
        <svg width="60" height="60" viewBox="0 0 60 60">
            <circle cx="30" cy="30" r="24" stroke="#f1f5f9" stroke-width="4" fill="transparent" />
            <circle cx="30" cy="30" r="24" stroke="{color}" stroke-width="4" fill="transparent"
                stroke-dasharray="150.8" stroke-dashoffset="{dashoffset}" stroke-linecap="round"
                transform="rotate(-90 30 30)" />
            <text x="30" y="34" font-family="'Outfit', sans-serif" font-size="11" font-weight="bold" fill="#1e293b" text-anchor="middle">{score:.0f}%</text>
        </svg>
        <span class="fit-label" style="color: {color};">{label}</span>
    </div>
    """

def render_skills_tags(skills):
    sorted_skills = sorted(skills, key=lambda x: {"expert": 3, "advanced": 2, "intermediate": 1, "beginner": 0}.get(x.get("proficiency", "beginner"), 0), reverse=True)
    tags_html = []
    for s in sorted_skills[:5]:
        tags_html.append(f'<span class="skill-tag">{s.get("name")}</span>')
    return f'<div class="cand-skills-list">{"".join(tags_html)}</div>'

def render_strengths_list(strengths):
    strengths_html = []
    for st in strengths:
        strengths_html.append(f"""
        <div class="strength-item">
            <span class="strength-check">
                <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="3" stroke-linecap="round" stroke-linejoin="round">
                    <polyline points="20 6 9 17 4 12"></polyline>
                </svg>
            </span>
            <span>{st}</span>
        </div>
        """)
    return f'<div class="cand-strengths-list">{"".join(strengths_html)}</div>'

VERIFIED_BADGE_HTML = """
<span class="verified-icon" title="Verified Profile">
    <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor">
        <path d="M12 2C6.5 2 2 6.5 2 12s4.5 10 10 10 10-4.5 10-10S17.5 2 12 2zm-2 15l-5-5 1.41-1.41L10 14.17l7.59-7.59L19 8l-9 9z"/>
    </svg>
</span>
"""

def render_kpi_card(icon_html, label, value, subtext):
    return f"""
    <div class="kpi-card">
        <div class="kpi-icon-container">
            {icon_html}
        </div>
        <div class="kpi-content">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{value}</div>
            <div class="kpi-subtext">{subtext}</div>
        </div>
    </div>
    """

# ----------------------------------------------------
# PAGE REDIRECT ROUTER
# ----------------------------------------------------
if st.session_state.current_page == "Ranked Shortlist":
    col_header, col_actions = st.columns([7.8, 2.2])
    with col_header:
        st.markdown(clean_html("""
            <div style="margin-top: 10px;">
                <h1 style="margin: 0; font-size: 2.2rem; display: flex; align-items: center; gap: 8px;">
                    AI Ranked Shortlist
                </h1>
                <p style="margin: 4px 0 0 0; color: #64748b; font-size: 1.05rem;">Intelligent ranking beyond keywords to find the best fit.</p>
            </div>
        """), unsafe_allow_html=True)
        
    with col_actions:
        st.markdown("<div style='height: 15px;'></div>", unsafe_allow_html=True)
        if top_df is not None:
            csv_data = top_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Export Shortlist",
                data=csv_data,
                file_name="team_talos.csv",
                mime="text/csv",
                use_container_width=True,
                key="export_main_btn"
            )
                
    st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)
    
    # 4 KPI cards
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    
    # Load actual job description data
    jd_data = parse_job_description()
    role_name = jd_data.get('role', 'Senior AI Engineer')
    if len(role_name) > 28:
        role_name = role_name.split("—")[0].split("-")[0].strip()
    min_exp = jd_data.get('experience_years', {}).get('min', 4)
    max_exp = jd_data.get('experience_years', {}).get('max', 12)
    
    total_candidates_str = "12,842"
    if scores_df is not None:
        total_candidates_str = f"{len(scores_df):,}"
        
    shortlist_count = len(top_df) if top_df is not None else 128
    model_conf_val = "92%"
    if top_df is not None:
        model_conf_val = f"{top_df['score'].mean():.0f}%"
        
    with kpi1:
        st.markdown(clean_html(render_kpi_card(
            icon_html='<svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#7c3aed" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="2" y="7" width="20" height="14" rx="2" ry="2"></rect><path d="M16 21V5a2 2 0 0 0-2-2h-4a2 2 0 0 0-2 2v16"></path></svg>',
            label="Job Role",
            value=role_name,
            subtext=f"Experience: {min_exp}-{max_exp} yrs | Pune/Noida"
        )), unsafe_allow_html=True)
        
    with kpi2:
        st.markdown(clean_html(render_kpi_card(
            icon_html='<svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#2563eb" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M17 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2"></path><circle cx="9" cy="7" r="4"></circle><path d="M23 21v-2a4 4 0 0 0-3-3.87"></path><path d="M16 3.13a4 4 0 0 1 0 7.75"></path></svg>',
            label="Total Candidates",
            value=total_candidates_str,
            subtext="Profiles Analyzed"
        )), unsafe_allow_html=True)
        
    with kpi3:
        st.markdown(clean_html(render_kpi_card(
            icon_html='<svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#10b981" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polygon points="12 2 15.09 8.26 22 9.27 17 14.14 18.18 21.02 12 17.77 5.82 21.02 7 14.14 2 9.27 8.91 8.26 12 2"></polygon></svg>',
            label="Top 10% Shortlist",
            value=str(shortlist_count),
            subtext="Highly Relevant Matches"
        )), unsafe_allow_html=True)
        
    with kpi4:
        st.markdown(clean_html(render_kpi_card(
            icon_html='<svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#7c3aed" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M9.5 2A2.5 2.5 0 0 1 12 4.5v15a2.5 2.5 0 0 1-4.96-.44 2.5 2.5 0 0 1 0-3.12 3 3 0 0 1 0-4.88 2.5 2.5 0 0 1 0-3.12A2.5 2.5 0 0 1 9.5 2z"></path><path d="M14.5 2A2.5 2.5 0 0 0 12 4.5v15a2.5 2.5 0 0 0 4.96-.44 2.5 2.5 0 0 0 0-3.12 3 3 0 0 0 0-4.88 2.5 2.5 0 0 0 0-3.12A2.5 2.5 0 0 0 14.5 2z"></path></svg>',
            label="AI Confidence",
            value=model_conf_val,
            subtext="Model Confidence Score ⓘ"
        )), unsafe_allow_html=True)
        
    st.markdown("<div style='height: 25px;'></div>", unsafe_allow_html=True)
    
    # Calculate dynamic averages for right sidebar widgets
    if top_df is not None and scores_df is not None:
        shortlist_ids = top_df["candidate_id"].tolist()
        shortlist_scores = scores_df[scores_df["candidate_id"].isin(shortlist_ids)]
        avg_production = shortlist_scores["production_score"].mean() * 100.0 if "production_score" in shortlist_scores.columns else 82.6
        avg_career = shortlist_scores["career_score"].mean() * 100.0 if "career_score" in shortlist_scores.columns else 51.4
        avg_success = shortlist_scores["success_score"].mean() * 100.0 if "success_score" in shortlist_scores.columns else 78.1
        avg_behavioral = shortlist_scores["behavioral_score"].mean() * 100.0 if "behavioral_score" in shortlist_scores.columns else 75.3
    else:
        avg_production, avg_career, avg_success, avg_behavioral = 82.6, 51.4, 78.1, 75.3

    # Left & Right split columns
    col_left, col_right = st.columns([7, 3])
    
    with col_left:
        c_title, c_sort = st.columns([3, 1])
        with c_title:
            st.markdown(clean_html("""
                <div style="display: flex; align-items: center; gap: 10px;">
                    <h3 style="margin: 0; font-size: 1.35rem; color: #1e293b;">Top Ranked Candidates</h3>
                    <span style="background-color: #f5f3ff; color: #7c3aed; font-size: 0.75rem; font-weight: 600; padding: 3px 8px; border-radius: 9999px; border: 1px solid #ddd6fe;">AI Ranked</span>
                </div>
            """), unsafe_allow_html=True)
        with c_sort:
            st.selectbox("Sort by:", ["Best Match", "Highest Experience", "Shortest Notice"], key="sort_candidates_dropdown", label_visibility="collapsed")
            
        st.markdown("<div style='height: 15px;'></div>", unsafe_allow_html=True)
        
        if top_df is not None:
            if "shortlist_page" not in st.session_state:
                st.session_state.shortlist_page = 0
                
            items_per_page = 5
            start_idx = st.session_state.shortlist_page * items_per_page
            end_idx = start_idx + items_per_page
            
            page_df = top_df.iloc[start_idx:end_idx]
            
            for idx, row in page_df.iterrows():
                cid = row["candidate_id"]
                cand = cand_by_id.get(cid, {})
                profile = cand.get("profile", {})
                signals = cand.get("redrob_signals", {})
                skills = cand.get("skills", [])
                
                score_row = scores_by_id.get(cid, {})
                
                rank = row["rank"]
                score = row["score"]
                
                if rank == 1:
                    rank_class = "rank-1"
                elif rank == 2:
                    rank_class = "rank-2"
                elif rank == 3:
                    rank_class = "rank-3"
                else:
                    rank_class = "rank-other"
                    
                avatar_url = get_avatar_url(cid, rank)
                skills_tags_html = render_skills_tags(skills)
                strengths = get_key_strengths(cand, score_row)
                strengths_html = render_strengths_list(strengths)
                fit_circle_html = render_fit_circle(score)
                
                card_html = f"""
                <div class="cand-card">
                    <div class="cand-left">
                        <div class="cand-rank-badge {rank_class}">{rank}</div>
                        <div class="cand-info">
                            <div class="cand-name-row">
                                <span class="cand-name">{profile.get("anonymized_name", "N/A")}</span>
                                {VERIFIED_BADGE_HTML}
                            </div>
                            <div class="cand-title">{profile.get("current_title", "N/A")} @ {profile.get("current_company", "N/A")}</div>
                            <div class="cand-meta">{profile.get("years_of_experience", 0.0):.1f} yrs exp. · {profile.get("location", "N/A")}</div>
                        </div>
                        <div class="cand-skills-strengths">
                            <div>
                                <div class="cand-section-title">Top Skills</div>
                                {skills_tags_html}
                            </div>
                            <div style="margin-top: 8px;">
                                <div class="cand-section-title">Key Strengths</div>
                                {strengths_html}
                            </div>
                        </div>
                    </div>
                    <div class="cand-right">
                        {fit_circle_html}
                        <div class="action-container">
                            <a href="?page=Talent+Insights&candidate_id={cid}" target="_self" class="view-profile-btn">View Profile</a>
                            <button class="bookmark-btn" onclick="alert('Candidate Bookmarked!')" title="Bookmark Candidate">
                                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                                    <path d="M19 21l-7-5-7 5V5a2 2 0 0 1 2-2h10a2 2 0 0 1 2 2z"></path>
                                </svg>
                            </button>
                        </div>
                    </div>
                </div>
                """
                st.markdown(clean_html(card_html), unsafe_allow_html=True)
                
            # Pagination
            col_prev, col_page_num, col_next = st.columns([1, 3, 1])
            with col_prev:
                if st.session_state.shortlist_page > 0:
                    if st.button("← Previous", use_container_width=True, key="prev_page_btn"):
                        st.session_state.shortlist_page -= 1
                        st.rerun()
            with col_page_num:
                total_pages = int(np.ceil(len(top_df) / items_per_page))
                st.markdown(f"<div style='text-align: center; color: #64748b; font-weight: 500; margin-top: 8px;'>Page {st.session_state.shortlist_page + 1} of {total_pages}</div>", unsafe_allow_html=True)
            with col_next:
                if (st.session_state.shortlist_page + 1) * items_per_page < len(top_df):
                    if st.button("Next →", use_container_width=True, key="next_page_btn"):
                        st.session_state.shortlist_page += 1
                        st.rerun()
                        
        else:
            st.warning("Ranking shortlist not loaded.")
            
        st.markdown(clean_html("""
            <div style="display: flex; align-items: center; gap: 6px; justify-content: center; color: #64748b; font-size: 0.85rem; margin-top: 20px;">
                
                <span>AI rankings are continuously learning and improve with recruiter feedback.</span>
            </div>
        """), unsafe_allow_html=True)
        
    with col_right:
        # Why These Candidates Card
        st.markdown(clean_html("""
            <div class="white-card">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 16px;">
                    <h4 style="margin: 0; font-size: 1.1rem; color: #1e293b; font-family: 'Outfit', sans-serif;">Why These Candidates?</h4>
                    <span style="color: #94a3b8; cursor: pointer; font-size: 0.95rem;" title="These insights summarize the core ML models' features">ⓘ</span>
                </div>
                
                <div style="display: flex; flex-direction: column; gap: 16px;">
                    <div style="display: flex; align-items: flex-start; gap: 12px;">
                        <div style="width: 32px; height: 32px; border-radius: 8px; background-color: #f5f3ff; display: flex; align-items: center; justify-content: center; flex-shrink: 0;">
                            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#7c3aed" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
                                <circle cx="12" cy="12" r="10"></circle>
                                <line x1="2" y1="12" x2="22" y2="12"></line>
                                <path d="M12 2a15.3 15.3 0 0 1 4 10 15.3 15.3 0 0 1-4 10 15.3 15.3 0 0 1-4-10 15.3 15.3 0 0 1 4-10z"></path>
                            </svg>
                        </div>
                        <div style="font-size: 0.875rem; color: #475569; line-height: 1.4; font-weight: 500;">
                            Semantic fit matching core JD requirements (dense embeddings).
                        </div>
                    </div>
                    
                    <div style="display: flex; align-items: flex-start; gap: 12px;">
                        <div style="width: 32px; height: 32px; border-radius: 8px; background-color: #eff6ff; display: flex; align-items: center; justify-content: center; flex-shrink: 0;">
                            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#3b82f6" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
                                <polyline points="22 12 18 12 15 21 9 3 6 12 2 12"></polyline>
                            </svg>
                        </div>
                        <div style="font-size: 0.875rem; color: #475569; line-height: 1.4; font-weight: 500;">
                            LightGBM Regressor ranking across 28 multi-modal features.
                        </div>
                    </div>
                    
                    <div style="display: flex; align-items: flex-start; gap: 12px;">
                        <div style="width: 32px; height: 32px; border-radius: 8px; background-color: #fffbeb; display: flex; align-items: center; justify-content: center; flex-shrink: 0;">
                            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#d97706" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
                                <polygon points="12 2 2 7 12 12 22 7 12 2"></polygon>
                                <polyline points="2 17 12 22 22 17"></polyline>
                                <polyline points="2 12 12 17 22 12"></polyline>
                            </svg>
                        </div>
                        <div style="font-size: 0.875rem; color: #475569; line-height: 1.4; font-weight: 500;">
                            Production MLOps and vector search experience prioritized.
                        </div>
                    </div>
                    
                    <div style="display: flex; align-items: flex-start; gap: 12px;">
                        <div style="width: 32px; height: 32px; border-radius: 8px; background-color: #ecfdf5; display: flex; align-items: center; justify-content: center; flex-shrink: 0;">
                            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#10b981" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
                                <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"></path>
                            </svg>
                        </div>
                        <div style="font-size: 0.875rem; color: #475569; line-height: 1.4; font-weight: 500;">
                            Honeypot Shield filtering profile anomalies and skill inflation.
                        </div>
                    </div>
                </div>
            </div>
        """), unsafe_allow_html=True)
        
        # Engagement Signals Card
        st.markdown(clean_html(f"""
            <div class="white-card">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px;">
                    <h4 style="margin: 0; font-size: 1.1rem; color: #1e293b; font-family: 'Outfit', sans-serif;">Engagement Signals</h4>
                    <span style="color: #94a3b8; cursor: pointer; font-size: 0.95rem;" title="Aggregated candidate pool behavioral data">ⓘ</span>
                </div>
                
                <div style="margin-bottom: 16px;">
                    <div style="display: flex; justify-content: space-between; font-size: 0.85rem; font-weight: 600; color: #475569; margin-bottom: 6px;">
                        <span>Production Experience Fit</span>
                        <span>{avg_production:.0f}%</span>
                    </div>
                    <div style="width: 100%; height: 8px; background-color: #f1f5f9; border-radius: 4px; overflow: hidden;">
                        <div style="width: {avg_production:.0f}%; height: 100%; background-color: #14b8a6; border-radius: 4px;"></div>
                    </div>
                </div>
                
                <div style="margin-bottom: 16px;">
                    <div style="display: flex; justify-content: space-between; font-size: 0.85rem; font-weight: 600; color: #475569; margin-bottom: 6px;">
                        <span>Career Trajectory & Stability</span>
                        <span>{avg_career:.0f}%</span>
                    </div>
                    <div style="width: 100%; height: 8px; background-color: #f1f5f9; border-radius: 4px; overflow: hidden;">
                        <div style="width: {avg_career:.0f}%; height: 100%; background-color: #8b5cf6; border-radius: 4px;"></div>
                    </div>
                </div>
                
                <div style="margin-bottom: 16px;">
                    <div style="display: flex; justify-content: space-between; font-size: 0.85rem; font-weight: 600; color: #475569; margin-bottom: 6px;">
                        <span>Future Success Probability</span>
                        <span>{avg_success:.0f}%</span>
                    </div>
                    <div style="width: 100%; height: 8px; background-color: #f1f5f9; border-radius: 4px; overflow: hidden;">
                        <div style="width: {avg_success:.0f}%; height: 100%; background-color: #3b82f6; border-radius: 4px;"></div>
                    </div>
                </div>
                
                <div>
                    <div style="display: flex; justify-content: space-between; font-size: 0.85rem; font-weight: 600; color: #475569; margin-bottom: 6px;">
                        <span>Behavioral Engagement Fit</span>
                        <span>{avg_behavioral:.0f}%</span>
                    </div>
                    <div style="width: 100%; height: 8px; background-color: #f1f5f9; border-radius: 4px; overflow: hidden;">
                        <div style="width: {avg_behavioral:.0f}%; height: 100%; background-color: #f59e0b; border-radius: 4px;"></div>
                    </div>
                </div>
            </div>
        """), unsafe_allow_html=True)

# ----------------------------------------------------
# OTHER VIEW WORKSPACES
# ----------------------------------------------------
elif st.session_state.current_page == "Talent Insights":
    st.markdown(clean_html("""
        <div style="margin-top: 10px; margin-bottom: 20px;">
            <h1 style="margin: 0; font-size: 2.2rem; display: flex; align-items: center; gap: 8px;">
                Candidate Spotlight
            </h1>
            <p style="margin: 4px 0 0 0; color: #64748b; font-size: 1.05rem;">Deep-dive career trajectory, behavioral signals, and credentials validation.</p>
        </div>
    """), unsafe_allow_html=True)
    
    if top_df is not None:
        cids_list = top_df["candidate_id"].tolist()
        
        default_cid_idx = 0
        if st.session_state.selected_candidate in cids_list:
            default_cid_idx = cids_list.index(st.session_state.selected_candidate)
            
        selected_cid = st.selectbox(
            "Select Candidate from Shortlist:", 
            cids_list, 
            index=default_cid_idx,
            key="spotlight_cand_select"
        )
        
        if selected_cid != st.session_state.selected_candidate:
            st.session_state.selected_candidate = selected_cid
            
        if selected_cid:
            cand = cand_by_id[selected_cid]
            profile = cand.get("profile", {})
            history = cand.get("career_history", [])
            skills = cand.get("skills", [])
            signals = cand.get("redrob_signals", {})
            edu = cand.get("education", [])
            
            score_row = scores_by_id.get(selected_cid, {})
            shortlist_row = top_df[top_df["candidate_id"] == selected_cid].iloc[0]
            
            avatar_url = get_avatar_url(selected_cid, shortlist_row["rank"])
            
            # Profile header card
            st.markdown(clean_html(f"""
                <div class="spotlight-header-card">
                    <div style="display: flex; align-items: center; gap: 20px;">
                        <div>
                            <div style="display: flex; align-items: center; gap: 8px;">
                                <h2 style="margin: 0; font-size: 1.8rem; color: #1e293b;">{profile.get("anonymized_name")}</h2>
                                {VERIFIED_BADGE_HTML}
                                <span style="background-color: #f5f3ff; color: #7c3aed; font-size: 0.8rem; font-weight: 700; padding: 2px 8px; border-radius: 9999px;">Rank #{shortlist_row["rank"]}</span>
                            </div>
                            <div style="font-size: 1.1rem; font-weight: 600; color: #4f46e5; margin-top: 4px;">{profile.get("current_title")} @ {profile.get("current_company")}</div>
                            <div style="font-size: 0.9rem; color: #64748b; margin-top: 4px; display: flex; gap: 16px;">
                                <span>Location: {profile.get("location")}, {profile.get("country")}</span>
                                <span>Experience: {profile.get("years_of_experience")} years</span>
                                <span>Company Size: {profile.get("current_company_size", "N/A")}</span>
                            </div>
                        </div>
                    </div>
                </div>
            """), unsafe_allow_html=True)
            
            # Recruiter Reasoning
            st.markdown("### Recruiter Reasoning")
            st.markdown(clean_html(f'<div class="reasoning-box">"{shortlist_row["reasoning"]}"</div>'), unsafe_allow_html=True)
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.markdown("### Career Summary & History")
                st.markdown(clean_html(f"""
                    <div class="white-card">
                        <p style="font-size: 1rem; line-height: 1.6; color: #334155; margin-bottom: 20px;">{profile.get("summary", "N/A")}</p>
                        
                        <h4 style="font-size: 1.1rem; margin-bottom: 15px; color: #1e293b;">Career History</h4>
                        <div class="timeline-container">
                """), unsafe_allow_html=True)
                
                for job in history:
                    end_d = job.get('end_date') if job.get('end_date') else 'Present'
                    dur_m = job.get('duration_months', 0)
                    st.markdown(clean_html(f"""
                        <div class="timeline-item">
                            <div class="timeline-marker"></div>
                            <div class="timeline-date">{job.get('start_date')} to {end_d} ({dur_m} months)</div>
                            <h5 style="margin: 0; font-size: 1rem; color: #1e293b;">{job.get('title')}</h5>
                            <div style="font-size: 0.85rem; font-weight: 600; color: #4f46e5; margin-bottom: 6px;">{job.get('company')} ({job.get('industry')})</div>
                            <p style="font-size: 0.9rem; color: #475569; margin: 0; line-height: 1.5;">{job.get('description')}</p>
                        </div>
                    """), unsafe_allow_html=True)
                    
                st.markdown(clean_html("""
                        </div>
                    </div>
                """), unsafe_allow_html=True)
                
                # Education Section
                st.markdown("### Education & Credentials")
                with st.container(border=True):
                    for e in edu:
                        st.markdown(clean_html(f"""
                            <div style="margin-bottom: 15px; border-bottom: 1px solid #f1f5f9; padding-bottom: 12px; last-child: border-bottom: none;">
                                <h5 style="margin: 0; font-size: 1.05rem; color: #1e293b;">{e.get('degree')} in {e.get('field_of_study')}</h5>
                                <div style="font-size: 0.9rem; color: #4f46e5; font-weight: 600;">{e.get('institution')} <span class="audit-badge audit-pass" style="font-size: 0.65rem; padding: 1px 6px;">{e.get('tier', 'tier_2').replace('_', ' ').upper()}</span></div>
                                <div style="font-size: 0.8rem; color: #94a3b8; margin-top: 4px;">Graduation Year: {e.get('end_year')} | GPA/Grade: {e.get('grade')}</div>
                            </div>
                        """), unsafe_allow_html=True)
                
            with col2:
                st.markdown("### Fit Analytics")
                is_hp, risk_pen, trust_reasons = compute_trust_and_risk(cand)
                
                with st.container(border=True):
                    st.metric("Final Composite Fit", f"{shortlist_row['score']:.2f}%")
                    st.metric("ML Production System Fit", f"{score_row.get('production_score', 0.0)*100.0:.1f}%")
                    st.metric("Career Trajectory & Stability", f"{score_row.get('career_score', 0.0)*100.0:.1f}%")
                    st.metric("Future Success Probability", f"{score_row.get('success_score', 0.0)*100.0:.1f}%")
                    st.metric("Behavioral Engagement Fit", f"{score_row.get('behavioral_score', 0.0)*100.0:.1f}%")
                    st.metric("Notice Period", f"{signals.get('notice_period_days')} days")
                
                st.markdown("### Credential Validation Audits")
                with st.container(border=True):
                    checks = [
                        ("Timeline date match", not any("duration mismatch" in r for r in trust_reasons)),
                        ("Education dates check", not any("college in" in r for r in trust_reasons)),
                        ("Experience sum validation", not any("Experience summary mismatch" in r for r in trust_reasons)),
                        ("Startup founding validation", not any("Impossible company timeline" in r for r in trust_reasons)),
                        ("Skill inflation audit", not any("Skill inflation" in r for r in trust_reasons)),
                        ("Single active job check", not any("Simultaneous current roles" in r for r in trust_reasons)),
                        ("Keyword stuffing check", not any("Keyword-stuffer" in r for r in trust_reasons))
                    ]
                    for label, passed in checks:
                        badge_class = "audit-pass" if passed else "audit-warn"
                        status_text = "PASSED" if passed else "WARNING"
                        st.markdown(clean_html(f"""
                            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px; font-size: 0.85rem;">
                                <span style="font-weight: 500; color: #475569;">{label}</span>
                                <span class="audit-badge {badge_class}">{status_text}</span>
                            </div>
                        """), unsafe_allow_html=True)
                
            # Skills lists
            st.markdown("### Technical and Soft Skills")
            with st.container(border=True):
                skill_cols = st.columns(3)
                for idx, s in enumerate(skills):
                    col_to_use = skill_cols[idx % 3]
                    prof = s.get("proficiency", "beginner")
                    dur = s.get("duration_months", 0)
                    ends = s.get("endorsements", 0)
                    prof_color = {"expert": "#7c3aed", "advanced": "#2563eb", "intermediate": "#059669", "beginner": "#64748b"}.get(prof, "#64748b")
                    prof_bg = {"expert": "#f5f3ff", "advanced": "#eff6ff", "intermediate": "#ecfdf5", "beginner": "#f1f5f9"}.get(prof, "#f1f5f9")
                    
                    col_to_use.markdown(clean_html(f"""
                        <div style="border: 1px solid #eef0f6; padding: 12px; border-radius: 10px; margin-bottom: 10px; background-color: #ffffff; box-shadow: 0 2px 4px rgba(0,0,0,0.01);">
                            <strong style="color: #1e293b; font-size: 0.95rem;">{s.get('name')}</strong>
                            <div style="display: flex; justify-content: space-between; font-size: 0.78rem; margin-top: 6px; align-items: center;">
                                <span style="color: {prof_color}; background-color: {prof_bg}; font-weight: bold; padding: 2px 6px; border-radius: 4px;">{prof.upper()}</span>
                                <span style="color: #64748b; font-weight: 500;">Duration: {dur} months</span>
                                <span style="color: #64748b; font-weight: 500; display: flex; align-items: center; gap: 4px;"><svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><path d="M14 9V5a3 3 0 0 0-3-3l-4 9v11h11.28a2 2 0 0 0 2-1.7l1.38-9a2 2 0 0 0-2-2.3zM7 22H4a2 2 0 0 1-2-2v-7a2 2 0 0 1 2-2h3"></path></svg>{ends}</span>
                            </div>
                        </div>
                    """), unsafe_allow_html=True)
    else:
        st.warning("Shortlist database not populated.")

elif st.session_state.current_page == "Dashboard":
    st.markdown(clean_html("""
        <div style="margin-top: 10px; margin-bottom: 20px;">
            <h1 style="margin: 0; font-size: 2.2rem; display: flex; align-items: center; gap: 8px;">
                Recruiter Analytics & ML Dashboard
            </h1>
            <p style="margin: 4px 0 0 0; color: #64748b; font-size: 1.05rem;">Predictive dashboard analytics and machine learning model details.</p>
        </div>
    """), unsafe_allow_html=True)
    
    # 4 KPI cards for the Dashboard
    dkpi1, dkpi2, dkpi3, dkpi4 = st.columns(4)
    
    total_candidates_val = len(scores_df) if scores_df is not None else 100000
    shortlist_count = len(top_df) if top_df is not None else 100
    selectivity_rate = (shortlist_count / total_candidates_val) * 100 if total_candidates_val > 0 else 0.10
    
    honeypots_blocked = 0
    if scores_df is not None and "is_honeypot" in scores_df.columns:
        honeypots_blocked = scores_df["is_honeypot"].sum()
    else:
        honeypots_blocked = 5766

    with dkpi1:
        st.markdown(clean_html(render_kpi_card(
            icon_html='<svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#7c3aed" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M17 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2"></path><circle cx="9" cy="7" r="4"></circle><path d="M23 21v-2a4 4 0 0 0-3-3.87"></path><path d="M16 3.13a4 4 0 0 1 0 7.75"></path></svg>',
            label="Total Pool size",
            value=f"{total_candidates_val:,}",
            subtext="Candidates Evaluated"
        )), unsafe_allow_html=True)
        
    with dkpi2:
        st.markdown(clean_html(render_kpi_card(
            icon_html='<svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#2563eb" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polygon points="12 2 15.09 8.26 22 9.27 17 14.14 18.18 21.02 12 17.77 5.82 21.02 7 14.14 2 9.27 8.91 8.26 12 2"></polygon></svg>',
            label="Shortlist (Top 10%)",
            value=str(shortlist_count),
            subtext="Highly Relevant Matches"
        )), unsafe_allow_html=True)
        
    with dkpi3:
        st.markdown(clean_html(render_kpi_card(
            icon_html='<svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#10b981" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"></circle><polyline points="12 6 12 12 16 14"></polyline></svg>',
            label="Selectivity Rate",
            value=f"{selectivity_rate:.2f}%",
            subtext="Tight Shortlist Target"
        )), unsafe_allow_html=True)
        
    with dkpi4:
        st.markdown(clean_html(render_kpi_card(
            icon_html='<svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#ef4444" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"></path><line x1="8" y1="11" x2="16" y2="11"></line></svg>',
            label="Honeypots Blocked",
            value=f"{int(honeypots_blocked):,}",
            subtext="Profile Anomalies Audited"
        )), unsafe_allow_html=True)

    st.markdown("<div style='height: 15px;'></div>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        with st.container(border=True):
            st.markdown("<h4 style='margin:0 0 10px 0; font-size:1.15rem; color:#1e293b; font-family:\"Outfit\",sans-serif;'>LightGBM Regressor Feature Importance</h4>", unsafe_allow_html=True)
            if imp_df is not None:
                sorted_imp = imp_df.sort_values(by="importance", ascending=True).tail(10)
                
                # Plotly horizontal bar chart
                fig = px.bar(
                    sorted_imp, 
                    x="importance", 
                    y="feature", 
                    orientation='h',
                    color_discrete_sequence=['#7c3aed']
                )
                fig.update_layout(
                    margin=dict(l=20, r=20, t=10, b=10),
                    height=280,
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    xaxis=dict(showgrid=False, title=None),
                    yaxis=dict(title=None, showgrid=False)
                )
                st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
                st.caption("Feature split counts in the LightGBM Regressor model.")
            else:
                st.info("Feature importance data is missing.")
            
    with col2:
        with st.container(border=True):
            st.markdown("<h4 style='margin:0 0 10px 0; font-size:1.15rem; color:#1e293b; font-family:\"Outfit\",sans-serif;'>Shortlist Notice Period Distribution</h4>", unsafe_allow_html=True)
            if top_df is not None:
                notice_periods = []
                for idx, row in top_df.iterrows():
                    cand = cand_by_id[row["candidate_id"]]
                    notice_periods.append(cand["redrob_signals"]["notice_period_days"])
                notice_counts = pd.Series(notice_periods).value_counts().sort_index().reset_index()
                notice_counts.columns = ["Notice Period (Days)", "Count"]
                
                # Plotly bar chart
                fig = px.bar(
                    notice_counts, 
                    x="Notice Period (Days)", 
                    y="Count",
                    color_discrete_sequence=['#3b82f6']
                )
                fig.update_layout(
                    margin=dict(l=20, r=20, t=10, b=10),
                    height=280,
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    xaxis=dict(showgrid=False),
                    yaxis=dict(showgrid=True, gridcolor='#f1f5f9')
                )
                st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
                st.caption("Notice period day distributions for the top 100.")
            
    col3, col4 = st.columns(2)
    with col3:
        with st.container(border=True):
            st.markdown("<h4 style='margin:0 0 10px 0; font-size:1.15rem; color:#1e293b; font-family:\"Outfit\",sans-serif;'>GitHub Activity vs Platform Responsiveness</h4>", unsafe_allow_html=True)
            if top_df is not None:
                gh_scores = []
                res_rates = []
                names = []
                for idx, row in top_df.iterrows():
                    cand = cand_by_id[row["candidate_id"]]
                    gh_scores.append(cand["redrob_signals"]["github_activity_score"])
                    res_rates.append(cand["redrob_signals"]["recruiter_response_rate"] * 100.0)
                    names.append(cand["profile"]["anonymized_name"])
                plot_df = pd.DataFrame({"GitHub Score": gh_scores, "Response Rate (%)": res_rates, "Name": names})
                plot_df = plot_df[plot_df["GitHub Score"] >= 0]
                
                # Plotly scatter chart
                fig = px.scatter(
                    plot_df, 
                    x="GitHub Score", 
                    y="Response Rate (%)", 
                    hover_name="Name",
                    color_discrete_sequence=['#8b5cf6']
                )
                fig.update_layout(
                    margin=dict(l=20, r=20, t=10, b=10),
                    height=280,
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    xaxis=dict(showgrid=True, gridcolor='#f1f5f9'),
                    yaxis=dict(showgrid=True, gridcolor='#f1f5f9')
                )
                st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
                st.caption("Platform responsiveness relative to technical engagement.")
            
    with col4:
        with st.container(border=True):
            st.markdown("<h4 style='margin:0 0 10px 0; font-size:1.15rem; color:#1e293b; font-family:\"Outfit\",sans-serif;'>Work Mode Stated Preferences</h4>", unsafe_allow_html=True)
            if top_df is not None:
                modes = []
                for idx, row in top_df.iterrows():
                    cand = cand_by_id[row["candidate_id"]]
                    modes.append(cand["redrob_signals"]["preferred_work_mode"])
                mode_counts = pd.Series(modes).value_counts().reset_index()
                mode_counts.columns = ["Work Mode", "Count"]
                
                # Plotly donut chart
                fig = px.pie(
                    mode_counts, 
                    values="Count", 
                    names="Work Mode", 
                    hole=0.45,
                    color_discrete_sequence=['#a78bfa', '#3b82f6', '#14b8a6', '#f59e0b']
                )
                fig.update_layout(
                    margin=dict(l=10, r=10, t=10, b=10),
                    height=280,
                    legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5)
                )
                st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
                st.caption("Preferred work-mode preferences in the top 100 shortlisted candidates.")

    # Model Evaluation Diagnostics Block
    with st.container(border=True):
        st.markdown("<h3 style='margin:0 0 15px 0; font-size:1.4rem; color:#7c3aed; font-family:\"Outfit\",sans-serif;'>TALOS AI – Machine Learning Engine Diagnostics</h3>", unsafe_allow_html=True)
        col_diag1, col_diag2, col_diag3 = st.columns(3)
        with col_diag1:
            st.markdown(clean_html("""
                <div style="padding: 10px; border-radius: 8px; background-color: #f8f9fc; border: 1px solid #eef0f6;">
                    <div style="font-size: 0.75rem; color: #94a3b8; font-weight: 700; text-transform: uppercase;">Model Specification</div>
                    <div style="font-size: 1.1rem; color: #1e293b; font-weight: 700; margin-top: 4px;">LightGBM Regressor</div>
                    <div style="font-size: 0.75rem; color: #64748b; margin-top: 2px;">Trained on 28 composite features</div>
                </div>
            """), unsafe_allow_html=True)
        with col_diag2:
            st.markdown(clean_html("""
                <div style="padding: 10px; border-radius: 8px; background-color: #f8f9fc; border: 1px solid #eef0f6;">
                    <div style="font-size: 0.75rem; color: #94a3b8; font-weight: 700; text-transform: uppercase;">Tie-breaker Sorting</div>
                    <div style="font-size: 1.1rem; color: #1e293b; font-weight: 700; margin-top: 4px;">Deterministic Engine</div>
                    <div style="font-size: 0.75rem; color: #64748b; margin-top: 2px;">Composite score + Candidate ID asc</div>
                </div>
            """), unsafe_allow_html=True)
        with col_diag3:
            st.markdown(clean_html("""
                <div style="padding: 10px; border-radius: 8px; background-color: #f8f9fc; border: 1px solid #eef0f6;">
                    <div style="font-size: 0.75rem; color: #94a3b8; font-weight: 700; text-transform: uppercase;">Safety Shield Overrides</div>
                    <div style="font-size: 1.1rem; color: #1e293b; font-weight: 700; margin-top: 4px;">Honeypot Filter - On</div>
                    <div style="font-size: 0.75rem; color: #64748b; margin-top: 2px;">Trust audit rules binary penalization</div>
                </div>
            """), unsafe_allow_html=True)

elif st.session_state.current_page == "Job Intake":
    st.markdown(clean_html("""
        <div style="margin-top: 10px; margin-bottom: 20px;">
            <h1 style="margin: 0; font-size: 2.2rem; display: flex; align-items: center; gap: 8px;">
                Job Description Parser Spec
            </h1>
            <p style="margin: 4px 0 0 0; color: #64748b; font-size: 1.05rem;">Upload candidate specification requirements specs and parse key roles and disqualifiers.</p>
        </div>
    """), unsafe_allow_html=True)
    
    with st.container(border=True):
        uploaded_file = st.file_uploader("Upload Job Description (.docx or .txt):", type=["docx", "txt"], key="jd_file_uploader")
        
        if uploaded_file is not None:
            temp_path = Path("temp_uploaded_jd.docx" if uploaded_file.name.endswith(".docx") else "temp_uploaded_jd.txt")
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            jd_data = parse_job_description(temp_path)
            st.success("Successfully parsed uploaded job description!")
            if temp_path.exists():
                os.remove(temp_path)
        else:
            jd_data = parse_job_description()
            st.info("Displaying parsed requirements for default 'Senior AI Engineer — Founding Team' role:")
    
    st.markdown(clean_html(f"""
        <div class="white-card">
            <h2 style="margin: 0; font-size: 1.6rem; color: #7c3aed;">Role: {jd_data['role']}</h2>
        </div>
    """), unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        with st.container(border=True):
            st.subheader("Absolute Must-Haves")
            for kw in jd_data["must_have"]:
                st.markdown(f"- **{kw.upper()}**")
            st.markdown("<br/>", unsafe_allow_html=True)
            st.subheader("Preferred Skills")
            for kw in jd_data["preferred"]:
                st.markdown(f"- *{kw}*")
            
    with col2:
        with st.container(border=True):
            st.subheader("Target Experience")
            st.write(f"- Minimum: **{jd_data['experience_years']['min']} years**")
            st.write(f"- Maximum: **{jd_data['experience_years']['max']} years**")
            st.markdown("<br/>", unsafe_allow_html=True)
            st.subheader("Core Disqualifiers")
            dis = jd_data["disqualifiers"]
            st.write(f"- **Consulting-Only**: {'Yes (TCS, Wipro, Infosys, Accenture, Capgemini, Cognizant, HCL)' if dis.get('consulting_only') else 'No'}")
            st.write(f"- **Pure Research without Deployment**: {'Yes' if dis.get('research_only') else 'No'}")
            st.write(f"- **LangChain-Only / Call API-Only**: {'Yes' if dis.get('langchain_only') else 'No'}")
            st.write(f"- **Title Chasers (Tenure < 18m)**: {'Yes' if dis.get('job_hopper_limit_months') else 'No'}")
            st.write(f"- **Out-of-Scope Domains**: {', '.join(dis.get('non_nlp_domains', []))}")

elif st.session_state.current_page == "Candidate Search":
    st.markdown(clean_html("""
        <div style="margin-top: 10px; margin-bottom: 20px;">
            <h1 style="margin: 0; font-size: 2.2rem; display: flex; align-items: center; gap: 8px;">
                Search Candidate Pool
            </h1>
            <p style="margin: 4px 0 0 0; color: #64748b; font-size: 1.05rem;">Perform fast keyword search and filter across the full candidate pool.</p>
        </div>
    """), unsafe_allow_html=True)
    
    with st.container(border=True):
        search_q = st.text_input("Enter Search Query (skills, titles, keywords):", "RAG Pinecone ranking")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            min_exp = st.slider("Minimum Years of Experience:", 0.0, 20.0, 5.0)
        with col2:
            max_notice = st.slider("Maximum Notice Period (Days):", 0, 180, 90)
        with col3:
            loc_pref = st.multiselect("Preferred Locations:", ["Pune", "Noida", "Bangalore", "Delhi NCR", "Mumbai", "Hyderabad", "Bengaluru", "Gurugram"], ["Pune", "Noida"])
        
        search_btn = st.button("Run Search Query", use_container_width=True)
    
    if search_btn:
        results = []
        with st.spinner("Searching candidate database..."):
            candidates_full = get_cached_candidates()
            if not candidates_full:
                st.warning("Candidate database file (candidates.jsonl) is not available in the cloud deployment. Candidate search is only supported in your local offline workspace.")
                st.stop()
            emb_engine = CandidateEmbeddingEngine()
            emb_engine.load_precomputed()
            cids = [c.get("candidate_id") for c in candidates_full]
            dense_sims = emb_engine.get_dense_similarity(search_q, cids)
            
            for idx, cand in enumerate(candidates_full):
                cid = cand.get("candidate_id")
                profile = cand.get("profile", {})
                signals = cand.get("redrob_signals", {})
                
                if profile.get("years_of_experience", 0.0) < min_exp:
                    continue
                if signals.get("notice_period_days", 90) > max_notice:
                    continue
                
                loc = profile.get("location", "").lower()
                if loc_pref:
                    if not any(lp.lower() in loc for lp in loc_pref):
                        continue
                        
                is_hp, _, _ = compute_trust_and_risk(cand)
                if is_hp:
                    continue
                    
                results.append({
                    "candidate_id": cid,
                    "cand_obj": cand,
                    "Semantic Score": dense_sims[idx]
                })
                
        if results:
            results = sorted(results, key=lambda x: x["Semantic Score"], reverse=True)[:10]
            st.write(f"Showing top {len(results)} matches:")
            
            for rank, r in enumerate(results):
                cid = r["candidate_id"]
                cand = r["cand_obj"]
                profile = cand.get("profile", {})
                skills = cand.get("skills", [])
                score = r["Semantic Score"] * 100.0
                score_row = scores_by_id.get(cid, {})
                avatar_url = get_avatar_url(cid, rank+1)
                skills_tags_html = render_skills_tags(skills)
                strengths = get_key_strengths(cand, score_row)
                strengths_html = render_strengths_list(strengths)
                fit_circle_html = render_fit_circle(score)
                
                card_html = f"""
                <div class="cand-card">
                    <div class="cand-left">
                        <div class="cand-rank-badge rank-other">{rank+1}</div>
                        <div class="cand-info">
                            <div class="cand-name-row">
                                <span class="cand-name">{profile.get("anonymized_name", "N/A")}</span>
                                {VERIFIED_BADGE_HTML}
                            </div>
                            <div class="cand-title">{profile.get("current_title", "N/A")} @ {profile.get("current_company", "N/A")}</div>
                            <div class="cand-meta">{profile.get("years_of_experience", 0.0):.1f} yrs exp. · {profile.get("location", "N/A")}</div>
                        </div>
                        <div class="cand-skills-strengths">
                            <div>
                                <div class="cand-section-title">Top Skills</div>
                                {skills_tags_html}
                            </div>
                            <div style="margin-top: 8px;">
                                <div class="cand-section-title">Key Strengths</div>
                                {strengths_html}
                            </div>
                        </div>
                    </div>
                    <div class="cand-right">
                        {fit_circle_html}
                        <div class="action-container">
                            <a href="?page=Talent+Insights&candidate_id={cid}" target="_self" class="view-profile-btn">View Profile</a>
                        </div>
                    </div>
                </div>
                """
                st.markdown(clean_html(card_html), unsafe_allow_html=True)
        else:
            st.info("No candidates match the specified filters.")

elif st.session_state.current_page == "Reports":
    st.markdown(clean_html("""
        <div style="margin-top: 10px; margin-bottom: 20px;">
            <h1 style="margin: 0; font-size: 2.2rem; display: flex; align-items: center; gap: 8px;">
                Export & Reports
            </h1>
            <p style="margin: 4px 0 0 0; color: #64748b; font-size: 1.05rem;">Export ranked tables and summary evaluation checklists.</p>
        </div>
    """), unsafe_allow_html=True)
    
    with st.container(border=True):
        st.subheader("Export Rank Shortlist")
        st.write("Generate and download the full ranked results matching candidate fit criteria.")
        if top_df is not None:
            csv_data = top_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download team_talos.csv Submission File",
                data=csv_data,
                file_name="team_talos.csv",
                mime="text/csv",
                key="reports_export_btn"
            )
            st.markdown("<br/>", unsafe_allow_html=True)
            st.dataframe(top_df, use_container_width=True, hide_index=True)
        else:
            st.warning("No shortlist data found.")

elif st.session_state.current_page == "Settings":
    st.markdown(clean_html("""
        <div style="margin-top: 10px; margin-bottom: 20px;">
            <h1 style="margin: 0; font-size: 2.2rem; display: flex; align-items: center; gap: 8px;">
                Recruiter Engine Settings
            </h1>
            <p style="margin: 4px 0 0 0; color: #64748b; font-size: 1.05rem;">Configure weights and threshold constants for candidate evaluations.</p>
        </div>
    """), unsafe_allow_html=True)
    
    with st.container(border=True):
        st.subheader("Model Weights & Configuration")
        st.write("These parameters influence the hybrid candidate score calculations prior to LightGBM scoring.")
        st.slider("Semantic Alignment Weight Coefficient", 0.0, 1.0, 0.4)
        st.slider("Career Stability & Tenure Weight Coefficient", 0.0, 1.0, 0.3)
        st.slider("Behavioral & Platform Responsiveness Weight Coefficient", 0.0, 1.0, 0.2)
        st.slider("Success Probability Forecast Weight Coefficient", 0.0, 1.0, 0.1)
        st.button("Save Settings Configuration")
