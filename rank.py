import argparse
import json
import csv
import pandas as pd
import numpy as np
from pathlib import Path
from src.utils import logger, load_candidates, find_data_dir
from src.embeddings import CandidateEmbeddingEngine
from src.ranking_model import CandidateRanker
from src.jd_parser import parse_job_description
from src.explainability import generate_reasoning
from src.trust_engine import compute_trust_and_risk

def main():
    parser = argparse.ArgumentParser(description="TALOS AI Candidate Ranking System")
    parser.add_argument("--candidates", type=str, required=True, help="Path to candidates.jsonl file")
    parser.add_argument("--out", type=str, required=True, help="Path to output submission CSV file")
    args = parser.parse_args()
    
    candidates_file = Path(args.candidates)
    out_file = Path(args.out)
    
    logger.info("=== TALOS AI CANDIDATE RANKING ENGINE ===")
    logger.info(f"Input Candidates File: {candidates_file}")
    logger.info(f"Output CSV File: {out_file}")
    
    # 1. Load Job Description & parse query text
    jd = parse_job_description()
    query_text = " ".join(jd["must_have"] + jd["preferred"])
    
    # 2. Load candidates list
    logger.info("Parsing candidates profiles...")
    # Load all candidates. We load them in memory.
    candidates = []
    with open(candidates_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                candidates.append(json.loads(line))
                
    logger.info(f"Loaded {len(candidates)} candidates.")
    
    # 3. Load embedding engine & precomputed files
    emb_engine = CandidateEmbeddingEngine()
    loaded_precomputed = emb_engine.load_precomputed()
    
    if not loaded_precomputed:
        logger.warning("Precomputed embeddings or vectorizer not found. Running on-the-fly precomputations...")
        emb_engine.precompute_and_save(candidates)
        
    # 4. Extract features
    ranker = CandidateRanker()
    df = ranker.extract_features_df(candidates, emb_engine, query_text)
    
    # 5. Score candidates
    logger.info("Scoring candidates using trained LightGBM model and applying Trust overrides...")
    final_scores = ranker.score_candidates(df)
    df["final_score"] = final_scores
    
    # 6. Rank candidates
    # We sort by: 1. Score (descending), 2. Candidate ID (ascending, for deterministic tiebreaking)
    df_sorted = df.sort_values(
        by=["final_score", "candidate_id"],
        ascending=[False, True]
    ).reset_index(drop=True)
    
    # 7. Take top 100 candidates and generate reasonings only for them (computationally fast)
    top_100_rows = df_sorted.head(100)
    top_100_data = []
    
    logger.info("Generating recruiter explanations for top-100 candidates...")
    
    # Re-map candidate list by ID for fast lookup
    cand_by_id = {c["candidate_id"]: c for c in candidates}
    
    ranking_explanations = {}
    
    for idx, row in top_100_rows.iterrows():
        cid = row["candidate_id"]
        score = row["final_score"]
        rank = idx + 1
        
        cand = cand_by_id[cid]
        
        # Recheck trust reasonings for explanation if any
        _, _, trust_reasons = compute_trust_and_risk(cand)
        
        # Generate reasoning
        reasoning = generate_reasoning(cand, score, row["semantic_fit"], row["production_score"], trust_reasons)
        
        top_100_data.append({
            "candidate_id": cid,
            "rank": rank,
            "score": round(score, 4),
            "reasoning": reasoning
        })
        
        ranking_explanations[cid] = {
            "rank": rank,
            "score": round(score, 4),
            "explanation": reasoning
        }
        
    # 8. Save submission CSV file
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["candidate_id", "rank", "score", "reasoning"])
        for row in top_100_data:
            writer.writerow([row["candidate_id"], row["rank"], f"{row['score']:.4f}", row["reasoning"]])
            
    logger.info(f"Submission CSV saved successfully to {out_file}.")
    
    # 9. Save support output files
    outputs_dir = find_data_dir() / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    
    # Save top_100_candidates.csv
    top_100_df = pd.DataFrame(top_100_data)
    top_100_df.to_csv(outputs_dir / "top_100_candidates.csv", index=False)
    logger.info(f"Top 100 CSV saved to {outputs_dir / 'top_100_candidates.csv'}")
    
    # Save ranking_explanations.json
    with open(outputs_dir / "ranking_explanations.json", "w", encoding="utf-8") as f:
        json.dump(ranking_explanations, f, indent=2)
    logger.info(f"Ranking explanations JSON saved to {outputs_dir / 'ranking_explanations.json'}")
    
    # Save top_100_profiles.json for instant Streamlit spotlight loading
    top_100_profiles = {cid: cand_by_id[cid] for cid in ranking_explanations.keys()}
    with open(outputs_dir / "top_100_profiles.json", "w", encoding="utf-8") as f:
        json.dump(top_100_profiles, f, indent=2)
    logger.info(f"Top 100 profiles JSON saved to {outputs_dir / 'top_100_profiles.json'}")
    
    # Save full candidate_scores.csv
    scores_df = df_sorted[["candidate_id", "final_score", "production_score", "career_score", "behavioral_score", "success_score", "is_honeypot"]]
    scores_df.to_csv(outputs_dir / "candidate_scores.csv", index=False)
    logger.info(f"All candidate scores saved to {outputs_dir / 'candidate_scores.csv'}")
    
    logger.info("=== RANKING SYSTEM RUN COMPLETED ===")

if __name__ == "__main__":
    main()
