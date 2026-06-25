from datetime import datetime

STARTUP_FOUNDING_YEARS = {
    "krutrim": 2023, "sarvam ai": 2023, "cred": 2018, "razorpay": 2014, "swiggy": 2014,
    "zomato": 2008, "flipkart": 2007, "paytm": 2010, "phonepe": 2015, "meesho": 2015,
    "freshworks": 2010, "dream11": 2008, "inmobi": 2007, "nykaa": 2012, "ola": 2010,
    "observe.ai": 2017, "rephrase.ai": 2019, "saarthi.ai": 2017, "verloop.io": 2015,
    "yellow.ai": 2016, "wysa": 2015, "haptik": 2013, "pied piper": 2014,
    "aganitha": 2017, "glance": 2019, "locobuzz": 2015, "mad street den": 2013,
    "niramai": 2016, "upgrad": 2015
}

NON_TECH_TITLES = {
    "accountant", "hr manager", "graphic designer", "content writer", "sales executive",
    "operations manager", "customer support", "marketing manager", "office manager"
}

def parse_date(date_str):
    if not date_str:
        return None
    try:
        return datetime.strptime(date_str, "%Y-%m-%d")
    except Exception:
        return None

def compute_trust_and_risk(cand):
    """
    Scans the candidate for anomalies and honeypot indicators.
    Returns (is_honeypot, risk_penalty, anomaly_reasons)
    """
    profile = cand.get("profile", {})
    history = cand.get("career_history", [])
    skills = cand.get("skills", [])
    edu = cand.get("education", [])
    
    is_honeypot = False
    reasons = []
    
    # 1. Career Started Before Education Started
    if edu and history:
        min_edu_start = min((e.get("start_year") for e in edu if e.get("start_year")), default=None)
        min_career_start = min((parse_date(j.get("start_date")).year for j in history if parse_date(j.get("start_date"))), default=None)
        if min_edu_start and min_career_start and min_career_start < min_edu_start - 2:
            is_honeypot = True
            reasons.append(f"Timeline clash: career started in {min_career_start} before college in {min_edu_start}")
            
    # 2. Job Duration vs Dates Mismatch
    for i, job in enumerate(history):
        start = parse_date(job.get("start_date"))
        end = parse_date(job.get("end_date"))
        duration_months = job.get("duration_months", 0)
        if start:
            # reference date: June 24, 2026
            actual_end = end if end else datetime(2026, 6, 24)
            diff_months = (actual_end.year - start.year) * 12 + (actual_end.month - start.month)
            if abs(duration_months - diff_months) > 24: # Allowing 2 years leeway, beyond that is fake
                is_honeypot = True
                reasons.append(f"Job {i+1} duration mismatch: dates indicate {diff_months} months, but duration is {duration_months}")
                
    # 3. Profile Experience vs History Sum Mismatch
    sum_months = sum(job.get("duration_months", 0) for job in history)
    profile_years = profile.get("years_of_experience", 0)
    if abs(profile_years - (sum_months / 12.0)) > 5.0:
        is_honeypot = True
        reasons.append(f"Experience summary mismatch: profile lists {profile_years} years, but work history sum is {sum_months/12.0:.1f} years")
        
    # 4. Startup Founding Date Anomaly
    for job in history:
        comp = job.get("company", "").lower().strip()
        start = parse_date(job.get("start_date"))
        if start:
            for sc, year in STARTUP_FOUNDING_YEARS.items():
                # check if company contains the startup name (handling slight variations)
                if sc == comp or (sc in comp and len(comp) < len(sc) + 5):
                    if start.year < year:
                        is_honeypot = True
                        reasons.append(f"Impossible company timeline: worked at {job.get('company')} in {start.year} (founded in {year})")
                        
    # 5. Skill Inflation (expert skills with 0 duration)
    expert_zero_dur = sum(1 for s in skills if s.get("proficiency") == "expert" and s.get("duration_months", 0) == 0)
    if expert_zero_dur >= 5:
        is_honeypot = True
        reasons.append(f"Skill inflation: {expert_zero_dur} expert skills listed with 0 months of usage")
        
    # 6. Multiple current full-time jobs
    current_jobs = [j for j in history if j.get("is_current")]
    if len(current_jobs) > 1:
        is_honeypot = True
        reasons.append(f"Simultaneous current roles: currently employed at multiple companies: {[j.get('company') for j in current_jobs]}")
        
    # 7. Keyword Stuffing / JD Trap Candidate
    current_title = profile.get("current_title", "").lower().strip()
    is_non_tech = current_title in NON_TECH_TITLES
    
    # Check if candidate has no tech-related title in their history
    history_titles = [j.get("title", "").lower() for j in history]
    has_tech_title = any(
        any(w in t for w in ["engineer", "developer", "scientist", "analyst", "tech", "programmer", "architect"])
        for t in history_titles
    )
    
    # If their current title is non-tech AND they have never held a tech title in their history,
    # but they have AI/ML core skills, it's a keyword stuffer trap!
    ml_keywords = {"embeddings", "retrieval", "vector", "ranking", "faiss", "milvus", "qdrant", "pinecone", "weaviate", "learning to rank", "lora", "fine-tuning", "rag", "search engine"}
    has_ml_skills = sum(1 for s in skills if any(k in s.get("name", "").lower() for k in ml_keywords)) >= 3
    
    if is_non_tech and not has_tech_title and has_ml_skills:
        # Penalize keyword stuffers heavily but don't strictly flag as honeypot (unless they fail other checks),
        # but apply the risk penalty to drop their score to 0.
        is_honeypot = True
        reasons.append(f"Keyword-stuffer trap: {current_title} profile with high ML skills but no technical career history")
        
    # Final penalty calculation
    risk_penalty = 100.0 if is_honeypot else 0.0
    
    return is_honeypot, risk_penalty, reasons
