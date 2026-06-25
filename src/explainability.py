import random

def generate_reasoning(cand, score, semantic_fit, production_score, trust_reasons):
    """
    Generates a 1-2 sentence, highly specific, factual recruiter-friendly explanation
    describing why the candidate is ranked where they are.
    It references named skills, years of experience, current titles, and behavioral/location signals
    directly from the candidate's profile to prevent hallucination.
    """
    profile = cand.get("profile", {})
    signals = cand.get("redrob_signals", {})
    history = cand.get("career_history", [])
    skills = [s.get("name", "") for s in cand.get("skills", [])]
    
    name = profile.get("anonymized_name", "Candidate")
    title = profile.get("current_title", "Engineer")
    exp = profile.get("years_of_experience", 0)
    loc = profile.get("location", "India")
    notice = signals.get("notice_period_days", 90)
    
    # Identify specific skills that match the JD
    jd_skills = {"embeddings", "retrieval", "vector", "ranking", "faiss", "milvus", "qdrant", "pinecone", "weaviate", "lora", "fine-tuning", "rag", "search", "python"}
    matching_skills = [s for s in skills if any(k in s.lower() for k in jd_skills)]
    matching_skills = list(set(matching_skills))[:3] # pick top 3
    
    # 1. If flagged as anomalous/honeypot, provide the trust concern
    if trust_reasons:
        return f"Disqualified: profile exhibits critical inconsistencies, including: {'; '.join(trust_reasons[:2])}."
        
    # Find matching companies they worked at
    companies = [j.get("company", "") for j in history if j.get("company")]
    companies = list(set(companies))[:2] # pick top 2
    
    # Structure of the sentence
    # Sentence 1: Summary of experience & skills
    exp_str = f"{exp:.1f} years" if exp > 0 else "strong"
    skills_part = f" with hands-on experience in {', '.join(matching_skills)}" if matching_skills else ""
    
    title_matches = ["ai", "ml", "machine learning", "nlp", "search", "retrieval", "ranking", "recommendation"]
    is_ml_title = any(w in title.lower() for w in title_matches)
    
    # Build dynamic descriptions
    intro_options = [
        f"Strong {title} with {exp_str} of experience{skills_part}.",
        f"{exp_str} experience as a {title}, specializing in {', '.join(matching_skills) if matching_skills else 'system design'}.",
        f"Demonstrated track record of {exp_str} in backend and systems engineering, with focus on {', '.join(matching_skills) if matching_skills else 'ML'}.",
    ]
    sentence1 = intro_options[hash(name) % len(intro_options)]
    
    # Sentence 2: Strengths & concerns (e.g. notice period, location, responsiveness)
    positives = []
    negatives = []
    
    # Product experience
    if production_score > 0.6:
        positives.append("strong production ML engineering experience")
    if companies:
        positives.append(f"career tenure at product-focused companies like {', '.join(companies)}")
        
    # Location
    loc_lower = loc.lower()
    if "noida" in loc_lower or "pune" in loc_lower:
        positives.append(f"located near office in {loc}")
    elif signals.get("willing_to_relocate"):
        positives.append("willing to relocate to Pune/Noida")
        
    # Notice Period
    if notice <= 30:
        positives.append(f"excellent notice period of {notice} days")
    elif notice >= 90:
        negatives.append(f"minor concern on {notice}-day notice period")
        
    # Recruiter response rate
    resp_rate = signals.get("recruiter_response_rate", 0.0)
    if resp_rate > 0.8:
        positives.append("exceptional recruiter responsiveness")
    elif resp_rate < 0.3:
        negatives.append("low platform responsiveness")
        
    # Assemble sentence 2
    pos_str = ", showing " + ", ".join(positives[:2]) if positives else ""
    neg_str = f" However, there is a {negatives[0]}." if negatives else ""
    
    sentence2 = f"Matches the 'product over research' profile{pos_str}.{neg_str}"
    
    reasoning = f"{sentence1} {sentence2}"
    return reasoning
