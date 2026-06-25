from datetime import datetime

PRESTIGIOUS_COMPANIES = {
    "google", "meta", "facebook", "microsoft", "amazon", "netflix", "apple", "uber", "salesforce", "adobe",
    "razorpay", "cred", "swiggy", "zomato", "flipkart", "inmobi", "freshworks", "phonepe", "dream11", "zoho"
}

CONSULTING_COMPANIES = {
    "tcs", "infosys", "wipro", "accenture", "cognizant", "capgemini", "hcl", "mphasis", "tech mahindra", "mindtree"
}

def parse_date(date_str):
    if not date_str:
        return None
    try:
        return datetime.strptime(date_str, "%Y-%m-%d")
    except Exception:
        return None

def compute_career_score(cand):
    """
    Evaluates career quality, promotion velocity, stability, and growth.
    Returns a score between 0.0 and 1.0.
    """
    profile = cand.get("profile", {})
    history = cand.get("career_history", [])
    
    if not history:
        return 0.0
        
    num_jobs = len(history)
    total_months = sum(job.get("duration_months", 0) for job in history)
    
    # 1. Average Tenure
    avg_tenure = total_months / num_jobs if num_jobs > 0 else 0
    # 36 months or more gets 1.0 tenure score. Below 12 months is heavily penalized.
    tenure_score = min(avg_tenure / 36.0, 1.0)
    
    # 2. Stability & Job Hopping Check
    # Stints under 18 months (excluding current jobs) are counted as hopping
    short_stints = 0
    for job in history:
        if not job.get("is_current") and job.get("duration_months", 0) < 18:
            short_stints += 1
            
    # Job hopping penalty
    hopping_penalty = min(short_stints * 0.15, 0.6)
    stability_score = max(tenure_score - hopping_penalty, 0.0)
    
    # 3. Promotions & Growth Velocity
    # Group jobs by company to find multiple titles (indicating promotions)
    comp_titles = {}
    for job in history:
        comp = job.get("company", "").strip().lower()
        title = job.get("title", "").strip().lower()
        if comp:
            if comp not in comp_titles:
                comp_titles[comp] = []
            comp_titles[comp].append(title)
            
    promotions = 0
    for comp, titles in comp_titles.items():
        if len(titles) >= 2:
            # Check if there is title change. Since they are different sequential titles in the same company, we count it as a promotion.
            unique_titles = len(set(titles))
            if unique_titles >= 2:
                promotions += (unique_titles - 1)
                
    promotion_score = min(promotions * 0.25, 1.0)
    
    # 4. Career Direction & Growth Rate
    # Did they transition from Consulting to Product/Tier-1?
    consulting_to_product = False
    has_product_exp = False
    
    # Order jobs from past to present to track company type transitions
    sorted_history = []
    for job in history:
        start_dt = parse_date(job.get("start_date"))
        if start_dt:
            sorted_history.append((start_dt, job))
    sorted_history.sort(key=lambda x: x[0])
    
    past_was_consulting = False
    for _, job in sorted_history:
        comp = job.get("company", "").strip().lower()
        is_consulting = any(c in comp for c in CONSULTING_COMPANIES)
        is_product = any(p in comp for p in PRESTIGIOUS_COMPANIES) or (not is_consulting and comp != "")
        
        if is_product:
            has_product_exp = True
            if past_was_consulting:
                consulting_to_product = True
        if is_consulting:
            past_was_consulting = True
            
    growth_score = 0.3 # base growth
    if has_product_exp:
        growth_score += 0.4
    if consulting_to_product:
        growth_score += 0.3
        
    # Title progression: mid to senior or staff
    current_title = profile.get("current_title", "").lower()
    if any(w in current_title for w in ["lead", "principal", "staff", "founding", "director", "head"]):
        growth_score = min(growth_score + 0.2, 1.0)
    elif "senior" in current_title or "sr" in current_title:
        growth_score = min(growth_score + 0.1, 1.0)
        
    # Combine scores
    career_score = 0.4 * stability_score + 0.3 * promotion_score + 0.3 * growth_score
    
    return min(max(career_score, 0.0), 1.0)
