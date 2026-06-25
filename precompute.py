import sys
import numpy as np
import pandas as pd
from pathlib import Path
from src.utils import logger, load_candidates, find_data_dir
from src.embeddings import CandidateEmbeddingEngine
from src.ranking_model import CandidateRanker
from src.jd_parser import parse_job_description

from src.production_engine import compute_production_score
from src.career_engine import compute_career_score
from src.behavioral_engine import compute_behavioral_signals
from src.trust_engine import compute_trust_and_risk
from src.scoring_engine import compute_location_fit

def main():
    logger.info("=== TALOS AI PRE-COMPUTATION PIPELINE ===")
    
    # 1. Load Job Description
    jd = parse_job_description()
    logger.info(f"Loaded Job Description for role: {jd['role']}")
    
    # Construct query text from JD must-have and preferred requirements
    query_text = " ".join(jd["must_have"] + jd["preferred"])
    logger.info(f"Query text representation: '{query_text}'")
    
    # 2. Load all candidates
    logger.info("Loading candidates from dataset...")
    candidates = list(load_candidates())
    logger.info(f"Total candidates loaded: {len(candidates)}")
    
    # 3. Run Fast Retrieval Stage
    logger.info("Step 1: Running Fast Retrieval Stage...")
    emb_engine = CandidateEmbeddingEngine()
    
    # Build TF-IDF matrix first
    tfidf_texts = [emb_engine.build_candidate_tfidf_text(c) for c in candidates]
    logger.info("Computing TF-IDF similarities...")
    tfidf_sims = emb_engine.get_tfidf_similarity(query_text, tfidf_texts)
    
    # Scale TF-IDF
    mn, mx = tfidf_sims.min(), tfidf_sims.max()
    norm_tfidf = (tfidf_sims - mn) / (mx - mn) if mx - mn > 1e-6 else tfidf_sims
    
    # Compute heuristic score for all candidates
    logger.info("Evaluating candidates with fast heuristics...")
    heuristic_scores = []
    
    consulting_firms = {"tcs", "infosys", "wipro", "accenture", "cognizant", "capgemini", "hcl", "mphasis", "tech mahindra", "mindtree"}
    
    for idx, cand in enumerate(candidates):
        is_hp, _, _ = compute_trust_and_risk(cand)
        history = cand.get("career_history", [])
        history_companies = [j.get("company", "").strip().lower() for j in history]
        is_consulting_only = all(any(c in comp for c in consulting_firms) for comp in history_companies) if history_companies else False
        
        # Honeypots and consulting-only candidates are filtered out from dense computation
        if is_hp or is_consulting_only:
            heuristic_scores.append(-1.0)
            continue
            
        prod_s = compute_production_score(cand)
        career_s = compute_career_score(cand)
        bh_signals = compute_behavioral_signals(cand)
        bh_s = bh_signals["hireability_score"]
        loc_s = compute_location_fit(cand)
        
        # Fast heuristic score combines key signals
        h_score = 0.35 * norm_tfidf[idx] + 0.25 * prod_s + 0.15 * bh_s + 0.15 * career_s + 0.10 * loc_s
        heuristic_scores.append(h_score)
        
    # Get top 8,000 candidate indices sorted by heuristic score
    sorted_indices = np.argsort(heuristic_scores)
    retrieve_indices = list(sorted_indices[-8000:])[::-1]
    
    logger.info(f"Retrieved top 8000 candidates for dense embeddings.")
    logger.info(f"Heuristic range: max={heuristic_scores[retrieve_indices[0]]:.4f}, min={heuristic_scores[retrieve_indices[-1]]:.4f}")
    
    # 4. Generate dense embeddings for the retrieved candidates
    logger.info("Step 2: Generating dense embeddings for retrieved candidates...")
    emb_engine.precompute_and_save(candidates, retrieve_indices)
    
    # 5. Feature Engineering
    logger.info("Step 3: Running Feature Engineering & Scoring Engine...")
    ranker = CandidateRanker()
    df = ranker.extract_features_df(candidates, emb_engine, query_text)
    
    # Save the feature dataframe temporarily for quick analysis if needed
    features_cache_path = find_data_dir() / "outputs" / "candidate_features.csv"
    features_cache_path.parent.mkdir(exist_ok=True)
    
    # Save scores to outputs
    df[["candidate_id", "target_score", "production_score", "is_honeypot"]].to_csv(
        find_data_dir() / "outputs" / "candidate_scores.csv", index=False
    )
    logger.info("Candidate scores saved to outputs.")
    
    # 6. Train LightGBM Model
    logger.info("Step 4: Training LightGBM Ranking Model...")
    ranker.train_model(df)
    
    logger.info("=== PRE-COMPUTATION COMPLETED SUCCESSFULLY ===")

if __name__ == "__main__":
    main()
