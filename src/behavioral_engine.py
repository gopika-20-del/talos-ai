from datetime import datetime

def parse_date(date_str):
    if not date_str:
        return None
    try:
        return datetime.strptime(date_str, "%Y-%m-%d")
    except Exception:
        return None

def compute_behavioral_signals(cand):
    """
    Evaluates behavioral signals from Redrob platform.
    Returns a dict with 'hireability_score' and 'recruiter_readiness'.
    """
    signals = cand.get("redrob_signals", {})
    
    # 1. Base components
    open_to_work = 1.0 if signals.get("open_to_work_flag") else 0.4
    
    # GitHub Activity
    gh_raw = signals.get("github_activity_score", -1)
    github_score = max(gh_raw, 0.0) / 100.0 if gh_raw != -1 else 0.1
    
    # Recruiter Response Rate
    response_rate = signals.get("recruiter_response_rate", 0.0)
    
    # Interview Completion Rate
    interview_completion = signals.get("interview_completion_rate", 0.0)
    
    # Offer Acceptance Rate
    offer_raw = signals.get("offer_acceptance_rate", -1)
    offer_acceptance = offer_raw if offer_raw != -1 else 0.6
    
    # Platform Recency (Active Date)
    active_date_str = signals.get("last_active_date")
    ref_date = datetime(2026, 6, 24) # Dataset reference date
    active_date = parse_date(active_date_str)
    
    if active_date:
        days_inactive = (ref_date - active_date).days
        if days_inactive <= 7:
            recency_score = 1.0
        elif days_inactive <= 30:
            recency_score = 0.9
        elif days_inactive <= 90:
            recency_score = 0.7
        elif days_inactive <= 180:
            recency_score = 0.4
        else:
            recency_score = 0.1
    else:
        recency_score = 0.5
        
    # Recruiter bookmarking & search views
    saved_30d = signals.get("saved_by_recruiters_30d", 0)
    views_30d = signals.get("profile_views_received_30d", 0)
    searches_30d = signals.get("search_appearance_30d", 0)
    
    saved_score = min(saved_30d / 10.0, 1.0)
    views_score = min(views_30d / 50.0, 1.0)
    searches_score = min(searches_30d / 500.0, 1.0)
    market_demand = 0.4 * saved_score + 0.3 * views_score + 0.3 * searches_score
    
    # 2. Hireability Score (predicts actual hiring success probability)
    hireability_score = (
        0.20 * open_to_work +
        0.15 * github_score +
        0.20 * response_rate +
        0.15 * interview_completion +
        0.15 * market_demand +
        0.15 * recency_score
    )
    
    # 3. Recruiter Readiness (how quickly they can be placed)
    notice_days = signals.get("notice_period_days", 90)
    if notice_days <= 15:
        notice_score = 1.0
    elif notice_days <= 30:
        notice_score = 0.9
    elif notice_days <= 60:
        notice_score = 0.6
    elif notice_days <= 90:
        notice_score = 0.3
    else: # 90+ days
        notice_score = 0.1
        
    # Profile verification trust score
    trust_elements = [
        signals.get("verified_email", False),
        signals.get("verified_phone", False),
        signals.get("linkedin_connected", False)
    ]
    trust_score = sum(1.0 for x in trust_elements if x) / 3.0
    # Add a baseline trust
    trust_score = 0.5 + 0.5 * trust_score
    
    recruiter_readiness = (
        0.50 * notice_score +
        0.30 * response_rate +
        0.20 * trust_score
    )
    
    return {
        "hireability_score": min(max(hireability_score, 0.0), 1.0),
        "recruiter_readiness": min(max(recruiter_readiness, 0.0), 1.0)
    }
