def compute_future_success_score(semantic_fit, production_score, career_score, behavioral_score, cand):
    """
    Predicts the probability that the candidate will succeed in the target role within 12 months.
    Returns a score between 0.0 and 1.0.
    """
    # Parse education tier factor
    edu = cand.get("education", [])
    tier_weights = {
        "tier_1": 1.0,
        "tier_2": 0.8,
        "tier_3": 0.5,
        "tier_4": 0.2,
        "unknown": 0.3
    }
    
    max_tier_weight = 0.3 # default if no edu
    if edu:
        max_tier_weight = max((tier_weights.get(e.get("tier", "unknown"), 0.3) for e in edu), default=0.3)
        
    # Boost if their current company is a Tier-1 product company
    current_company = cand.get("profile", {}).get("current_company", "").lower()
    prestigious_product_companies = {
        "google", "meta", "microsoft", "amazon", "netflix", "apple", "uber", "salesforce", "adobe",
        "razorpay", "cred", "swiggy", "zomato", "flipkart", "inmobi", "freshworks", "phonepe", "dream11", "zoho"
    }
    company_boost = 0.05 if any(p in current_company for p in prestigious_product_companies) else 0.0
    
    success_probability = (
        0.25 * semantic_fit +
        0.35 * production_score +
        0.20 * career_score +
        0.15 * behavioral_score +
        0.05 * max_tier_weight
    ) + company_boost
    
    return min(max(success_probability, 0.0), 1.0)
