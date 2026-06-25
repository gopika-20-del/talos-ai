import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from lightgbm import LGBMRegressor
from src.utils import logger, find_data_dir

# Import other engine functions
from src.production_engine import compute_production_score
from src.career_engine import compute_career_score
from src.behavioral_engine import compute_behavioral_signals
from src.trust_engine import compute_trust_and_risk
from src.success_predictor import compute_future_success_score
from src.scoring_engine import compute_location_fit, compute_hybrid_score

FEATURE_NAMES = [
    "semantic_fit",
    "tfidf_fit",
    "years_of_experience",
    "num_jobs",
    "avg_tenure",
    "promotion_count",
    "short_stints",
    "open_to_work",
    "github_score",
    "response_rate",
    "interview_completion",
    "saved_by_recruiters_30d",
    "profile_views_30d",
    "search_appearance_30d",
    "offer_acceptance_rate",
    "notice_period_days",
    "verified_email",
    "verified_phone",
    "linkedin_connected",
    "willing_to_relocate",
    "location_fit",
    "max_edu_tier",
    "has_consulting_only",
    "production_score",
    "career_score",
    "behavioral_score",
    "success_score",
    "readiness_score"
]

class CandidateRanker:
    def __init__(self):
        self.model_path = Path("./lightgbm_ranker.joblib")
        self.model = None
        
    def extract_features_df(self, candidates_list, emb_engine, query_text):
        """
        Extracts feature vectors for all candidates in the list and returns a DataFrame.
        """
        logger.info("Extracting feature vectors for training/scoring...")
        
        # 1. Get embedding similarities in bulk
        cids = [c.get("candidate_id") for c in candidates_list]
        dense_sims = emb_engine.get_dense_similarity(query_text, cids)
        tfidf_texts = [emb_engine.build_candidate_tfidf_text(c) for c in candidates_list]
        tfidf_sims = emb_engine.get_tfidf_similarity(query_text, tfidf_texts)
        
        # Scale dense and tfidf
        def min_max_scale(x):
            mn, mx = x.min(), x.max()
            if mx - mn < 1e-6:
                return np.zeros_like(x)
            return (x - mn) / (mx - mn)
            
        norm_dense = min_max_scale(dense_sims)
        norm_tfidf = min_max_scale(tfidf_sims)
        
        data = []
        
        # Set up mapping for edu tier
        tier_weights = {"tier_1": 1.0, "tier_2": 0.8, "tier_3": 0.5, "tier_4": 0.2, "unknown": 0.3}
        consulting_firms = {"tcs", "infosys", "wipro", "accenture", "cognizant", "capgemini", "hcl", "mphasis", "tech mahindra", "mindtree"}
        
        for idx, cand in enumerate(candidates_list):
            cid = cand.get("candidate_id")
            
            profile = cand.get("profile", {})
            history = cand.get("career_history", [])
            skills = cand.get("skills", [])
            signals = cand.get("redrob_signals", {})
            edu = cand.get("education", [])
            
            # Sub-scores
            prod_s = compute_production_score(cand)
            career_s = compute_career_score(cand)
            behavior_signals = compute_behavioral_signals(cand)
            bh_s = behavior_signals["hireability_score"]
            readiness_s = behavior_signals["recruiter_readiness"]
            loc_s = compute_location_fit(cand)
            
            sem_fit = norm_dense[idx]
            tfidf_fit = norm_tfidf[idx]
            
            success_s = compute_future_success_score(sem_fit, prod_s, career_s, bh_s, cand)
            
            # Simple aggregations
            num_jobs = len(history)
            total_months = sum(j.get("duration_months", 0) for j in history)
            avg_tenure = total_months / num_jobs if num_jobs > 0 else 0
            
            # Promotion count
            comp_titles = {}
            for job in history:
                comp = job.get("company", "").strip().lower()
                title = job.get("title", "").strip().lower()
                if comp:
                    if comp not in comp_titles:
                        comp_titles[comp] = []
                    comp_titles[comp].append(title)
            promotions = sum(max(len(set(titles)) - 1, 0) for comp, titles in comp_titles.items())
            
            # Short stints
            short_stints = sum(1 for j in history if not j.get("is_current") and j.get("duration_months", 0) < 18)
            
            # Education tier
            max_edu = max((tier_weights.get(e.get("tier", "unknown"), 0.3) for e in edu), default=0.3)
            
            # Consulting check
            history_companies = [j.get("company", "").strip().lower() for j in history]
            is_consulting_only = all(any(c in comp for c in consulting_firms) for comp in history_companies) if history_companies else False
            
            # GitHub activity
            gh_raw = signals.get("github_activity_score", -1)
            gh_score = max(gh_raw, 0.0) / 100.0 if gh_raw != -1 else 0.1
            
            # Verified and relocation
            relocate = 1.0 if signals.get("willing_to_relocate") else 0.0
            v_email = 1.0 if signals.get("verified_email") else 0.0
            v_phone = 1.0 if signals.get("verified_phone") else 0.0
            li_connected = 1.0 if signals.get("linkedin_connected") else 0.0
            open_work = 1.0 if signals.get("open_to_work_flag") else 0.0
            
            features = {
                "semantic_fit": sem_fit,
                "tfidf_fit": tfidf_fit,
                "years_of_experience": profile.get("years_of_experience", 0),
                "num_jobs": num_jobs,
                "avg_tenure": avg_tenure,
                "promotion_count": promotions,
                "short_stints": short_stints,
                "open_to_work": open_work,
                "github_score": gh_score,
                "response_rate": signals.get("recruiter_response_rate", 0.0),
                "interview_completion": signals.get("interview_completion_rate", 0.0),
                "saved_by_recruiters_30d": signals.get("saved_by_recruiters_30d", 0),
                "profile_views_30d": signals.get("profile_views_received_30d", 0),
                "search_appearance_30d": signals.get("search_appearance_30d", 0),
                "offer_acceptance_rate": signals.get("offer_acceptance_rate", 0.0) if signals.get("offer_acceptance_rate", -1) != -1 else 0.6,
                "notice_period_days": signals.get("notice_period_days", 90),
                "verified_email": v_email,
                "verified_phone": v_phone,
                "linkedin_connected": li_connected,
                "willing_to_relocate": relocate,
                "location_fit": loc_s,
                "max_edu_tier": max_edu,
                "has_consulting_only": 1.0 if is_consulting_only else 0.0,
                "production_score": prod_s,
                "career_score": career_s,
                "behavioral_score": bh_s,
                "success_score": success_s,
                "readiness_score": readiness_s
            }
            
            # Trust and anomalies
            is_honeypot, risk_penalty, reasons = compute_trust_and_risk(cand)
            features["is_honeypot"] = 1.0 if is_honeypot else 0.0
            features["risk_penalty"] = risk_penalty
            
            # Ground truth silver standard target
            features["target_score"] = compute_hybrid_score(
                sem_fit, prod_s, career_s, bh_s, success_s, readiness_s, loc_s, risk_penalty
            )
            
            data.append(features)
            
        df = pd.DataFrame(data)
        df["candidate_id"] = cids
        return df
        
    def train_model(self, df):
        """
        Trains a LightGBM Regressor using features and silver-standard target labels.
        """
        logger.info("Training LightGBM Regressor...")
        
        # Exclude target and candidate identifiers from training features
        # We also exclude target_score and risk_penalty, is_honeypot to let the model learn general fit
        X = df[FEATURE_NAMES]
        y = df["target_score"]
        
        self.model = LGBMRegressor(
            n_estimators=100,
            learning_rate=0.08,
            max_depth=6,
            num_leaves=31,
            random_state=42,
            verbose=-1
        )
        
        self.model.fit(X, y)
        
        logger.info(f"Saving trained model to {self.model_path}")
        joblib.dump(self.model, self.model_path)
        
        # Compute feature importance
        importances = self.model.feature_importances_
        imp_df = pd.DataFrame({
            "feature": FEATURE_NAMES,
            "importance": importances
        }).sort_values(by="importance", ascending=False)
        
        # Save to outputs directory
        out_dir = find_data_dir() / "outputs"
        out_dir.mkdir(exist_ok=True)
        imp_df.to_csv(out_dir / "feature_importance.csv", index=False)
        logger.info(f"Feature importance saved to {out_dir / 'feature_importance.csv'}")
        
    def load_model(self):
        """
        Loads the trained LightGBM model.
        """
        if self.model_path.exists():
            logger.info(f"Loading LightGBM model from {self.model_path.resolve()}")
            self.model = joblib.load(self.model_path)
            return True
        logger.warning("LightGBM model not found!")
        return False
        
    def score_candidates(self, df):
        """
        Scores the candidates using the loaded LightGBM model and applies Trust filters.
        """
        if self.model is None:
            if not self.load_model():
                raise ValueError("Model is not loaded or trained.")
                
        X = df[FEATURE_NAMES]
        
        # Predict base scores
        predicted_scores = self.model.predict(X)
        
        # Apply strict Trust & Risk override
        # If candidate is a honeypot or consulting_only, set score to 0.0
        final_scores = []
        for i, row in df.iterrows():
            cid = row["candidate_id"]
            pred = predicted_scores[i]
            
            # Read honeypot and consulting status
            is_hp = row["is_honeypot"] > 0.5
            is_consulting = row["has_consulting_only"] > 0.5
            
            if is_hp or is_consulting:
                final_score = 0.0
            else:
                # Clip prediction to [0, 100]
                final_score = min(max(pred, 0.0), 100.0)
                
            final_scores.append(final_score)
            
        return final_scores
