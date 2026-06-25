import os
import numpy as np
import joblib
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
from src.utils import logger, find_data_dir

class CandidateEmbeddingEngine:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = None
        self.tfidf_vectorizer = None
        self.candidate_embeddings_dense = None
        self.dense_candidate_ids = []
        self.candidate_tfidf_matrix = None
        self.vectorizer_path = Path("./tfidf_vectorizer.joblib")
        self.embeddings_path = Path("./candidate_embeddings.npz")
        
    def load_model(self):
        """
        Loads the SentenceTransformer model.
        """
        if self.model is None:
            logger.info(f"Loading SentenceTransformer model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
        return self.model
        
    def build_candidate_dense_text(self, cand):
        """
        Builds a compact string representation focusing on core skills, headline, and summary for dense embeddings.
        """
        profile = cand.get("profile", {})
        skills = cand.get("skills", [])
        
        parts = []
        if profile.get("headline"):
            parts.append(profile["headline"])
        if profile.get("summary"):
            parts.append(profile["summary"])
            
        skill_names = [s.get("name", "") for s in skills if s.get("name")]
        if skill_names:
            parts.append("Skills: " + ", ".join(skill_names))
            
        if profile.get("current_title"):
            parts.append(f"Current Role: {profile['current_title']}")
            
        return " ".join(parts)

    def build_candidate_tfidf_text(self, cand):
        """
        Builds a complete representation including full job descriptions for keyword searching.
        """
        profile = cand.get("profile", {})
        skills = cand.get("skills", [])
        history = cand.get("career_history", [])
        
        parts = [self.build_candidate_dense_text(cand)]
        
        for i, job in enumerate(history):
            title = job.get("title", "")
            comp = job.get("company", "")
            desc = job.get("description", "")
            job_str = f"Job {i+1}: {title} at {comp}."
            if desc:
                job_str += f" Details: {desc}"
            parts.append(job_str)
            
        return " ".join(parts)
        
    def precompute_and_save(self, candidates_list, retrieve_indices=None):
        """
        Generates and saves dense embeddings for the retrieved subset, and fits the TF-IDF vectorizer for all.
        """
        logger.info("Building TF-IDF text representations for all candidates...")
        tfidf_texts = [self.build_candidate_tfidf_text(c) for c in candidates_list]
        
        # 1. TF-IDF
        logger.info("Fitting and computing TF-IDF vectorizer...")
        self.tfidf_vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_features=10000)
        tfidf_matrix = self.tfidf_vectorizer.fit_transform(tfidf_texts)
        
        logger.info(f"Saving TF-IDF vectorizer to {self.vectorizer_path}")
        joblib.dump(self.tfidf_vectorizer, self.vectorizer_path)
        self.candidate_tfidf_matrix = tfidf_matrix
        
        # 2. Dense Embeddings (only for retrieved candidate subset)
        if retrieve_indices is None:
            # Fallback to all if not specified (will be slow)
            retrieve_indices = list(range(len(candidates_list)))
            
        logger.info(f"Computing dense embeddings for {len(retrieve_indices)} selected candidates...")
        selected_candidates = [candidates_list[idx] for idx in retrieve_indices]
        selected_dense_texts = [self.build_candidate_dense_text(c) for c in selected_candidates]
        selected_ids = [c.get("candidate_id") for c in selected_candidates]
        
        self.load_model()
        dense_embs = self.model.encode(
            selected_dense_texts, 
            batch_size=128, 
            show_progress_bar=True, 
            normalize_embeddings=True
        )
        
        # Save compressed dense embeddings along with their candidate IDs
        logger.info(f"Saving dense embeddings to {self.embeddings_path}")
        np.savez_compressed(
            self.embeddings_path, 
            embeddings=dense_embs, 
            candidate_ids=np.array(selected_ids)
        )
        self.candidate_embeddings_dense = dense_embs
        self.dense_candidate_ids = selected_ids
        
        return dense_embs, tfidf_matrix
        
    def load_precomputed(self):
        """
        Loads precomputed dense embeddings and TF-IDF vectorizer if available.
        """
        loaded = True
        
        # Load dense embeddings
        if self.embeddings_path.exists():
            logger.info(f"Loading dense embeddings from {self.embeddings_path.resolve()}")
            data = np.load(self.embeddings_path)
            self.candidate_embeddings_dense = data["embeddings"]
            self.dense_candidate_ids = list(data["candidate_ids"])
        else:
            logger.warning("Dense embeddings file not found!")
            loaded = False
            
        # Load TF-IDF vectorizer
        if self.vectorizer_path.exists():
            logger.info(f"Loading TF-IDF vectorizer from {self.vectorizer_path.resolve()}")
            self.tfidf_vectorizer = joblib.load(self.vectorizer_path)
        else:
            logger.warning("TF-IDF vectorizer file not found!")
            loaded = False
            
        return loaded
        
    def get_dense_similarity(self, query_text, candidate_ids):
        """
        Computes dense cosine similarity. For candidates in the precomputed set, returns actual score.
        For others, returns 0.0.
        """
        similarities = np.zeros(len(candidate_ids))
        
        if self.candidate_embeddings_dense is None:
            raise ValueError("Dense candidate embeddings are not loaded.")
            
        self.load_model()
        query_emb = self.model.encode([query_text], normalize_embeddings=True)[0]
        
        # Compute dot product for all loaded embeddings
        loaded_sims = np.dot(self.candidate_embeddings_dense, query_emb)
        
        # Map back to query list by candidate ID
        id_to_sim = dict(zip(self.dense_candidate_ids, loaded_sims))
        
        for idx, cid in enumerate(candidate_ids):
            similarities[idx] = id_to_sim.get(cid, 0.0)
            
        return similarities
        
    def get_tfidf_similarity(self, query_text, candidate_texts=None):
        """
        Computes TF-IDF similarity.
        """
        if self.tfidf_vectorizer is None:
            if self.vectorizer_path.exists():
                self.tfidf_vectorizer = joblib.load(self.vectorizer_path)
            else:
                if candidate_texts is not None:
                    logger.info("Fitting TF-IDF on-the-fly...")
                    self.tfidf_vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_features=10000)
                    self.candidate_tfidf_matrix = self.tfidf_vectorizer.fit_transform(candidate_texts)
                else:
                    raise ValueError("TF-IDF Vectorizer not loaded, and candidate texts not provided.")
                    
        query_tfidf = self.tfidf_vectorizer.transform([query_text])
        
        if self.candidate_tfidf_matrix is None and candidate_texts is not None:
            self.candidate_tfidf_matrix = self.tfidf_vectorizer.transform(candidate_texts)
            
        if self.candidate_tfidf_matrix is not None:
            sims = np.array(self.candidate_tfidf_matrix.dot(query_tfidf.T).todense()).flatten()
            return sims
        return np.zeros(len(candidate_texts))
