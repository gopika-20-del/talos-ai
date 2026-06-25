def compute_location_fit(cand):
    """
    Computes location fit based on Pune/Noida offices, welcome cities,
    and willingness to relocate. Returns a score between 0.0 and 1.0.
    """
    profile = cand.get("profile", {})
    signals = cand.get("redrob_signals", {})
    
    loc = profile.get("location", "").lower().strip()
    country = profile.get("country", "").lower().strip()
    willing_to_relocate = signals.get("willing_to_relocate", False)
    
    # 1. Check if outside India
    is_india = "india" in country or loc == "india"
    if not is_india and country != "":
        # Cases where country is not specified but location lists Indian cities
        indian_cities = ["noida", "pune", "delhi", "mumbai", "hyderabad", "bangalore", "chennai", "kolkata", "gurgaon"]
        if not any(c in loc for c in indian_cities):
            return 0.1 # Outside India, no visa sponsorship
            
    # 2. Check primary office locations
    if "noida" in loc or "pune" in loc:
        return 1.0
        
    # 3. Check welcome cities (Delhi NCR, Hyderabad, Mumbai)
    welcome_cities = ["delhi", "ncr", "gurgaon", "ghaziabad", "faridabad", "hyderabad", "mumbai"]
    is_welcome_city = any(c in loc for c in welcome_cities)
    
    # 4. Check other Tier-1 Indian cities
    tier1_cities = ["bangalore", "bengaluru", "chennai", "kolkata", "ahmedabad"]
    is_tier1 = any(c in loc for c in tier1_cities)
    
    if is_welcome_city:
        return 0.9 if willing_to_relocate else 0.7
    elif is_tier1:
        return 0.8 if willing_to_relocate else 0.5
    else:
        # Other Indian cities
        return 0.6 if willing_to_relocate else 0.3

def compute_hybrid_score(semantic_fit, production_score, career_score, hireability_score, success_score, readiness_score, location_fit, risk_penalty):
    """
    Computes the final raw score based on the hackathon guidelines:
    final_score = (
        0.35 * semantic_fit +
        0.20 * production_score +
        0.15 * hireability_score +
        0.10 * career_score +
        0.10 * success_score +
        0.05 * readiness_score +
        0.05 * location_fit
    ) - risk_penalty
    
    Normalizes the final score to be on 0 to 100 range.
    """
    raw_score = (
        0.35 * semantic_fit +
        0.20 * production_score +
        0.15 * hireability_score +
        0.10 * career_score +
        0.10 * success_score +
        0.05 * readiness_score +
        0.05 * location_fit
    ) - risk_penalty
    
    # Normalizing raw score (which is in range [-100, 1.0]) to [0, 100]
    # We clip raw_score to [0, 1.0] if there is no risk penalty, then scale by 100.
    # If there is a risk penalty, score will be <= 0.
    normalized = max(raw_score, 0.0) * 100.0
    return normalized
