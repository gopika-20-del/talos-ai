import re

# Keyword lists for matching production experiences
VEC_DBS = {"pinecone", "weaviate", "qdrant", "milvus", "faiss", "chromadb", "elasticsearch", "opensearch"}
PROD_KEYWORDS = {"production", "deploy", "scale", "infrastructure", "latency", "monitoring", "serving", "pipeline", "real-time", "kubernetes", "docker", "mlops", "aws", "gcp", "azure", "cicd", "throughput", "ab test", "a/b test"}
RETRIEVAL_KEYWORDS = {"retrieval", "vector search", "hybrid search", "dense search", "sparse search", "bm25", "information retrieval", "rag", "embeddings", "sentence-transformers"}
RANKING_KEYWORDS = {"ranking", "ranker", "recommendation", "recommender", "learning to rank", "ltr", "reranking", "collaborative filtering"}
EVAL_KEYWORDS = {"ndcg", "mrr", "map", "evaluation framework", "evaluation infrastructure", "offline evaluation", "offline benchmark"}
LORA_KEYWORDS = {"lora", "qlora", "peft", "fine-tuning", "fine-tune"}

TITLE_WEIGHTED_WORDS = {"ai", "ml", "machine learning", "nlp", "search", "retrieval", "ranking", "recommendation", "data scientist", "deep learning"}

def compute_production_score(cand):
    """
    Evaluates the candidate's career descriptions and skills for production ML experience.
    Returns a score between 0.0 and 1.0.
    """
    history = cand.get("career_history", [])
    skills = [s.get("name", "").lower() for s in cand.get("skills", [])]
    
    # 1. Skill checklist
    skills_set = set(skills)
    has_vec_db = any(db in skills_set for db in VEC_DBS)
    has_prod = any(p in skills_set for p in PROD_KEYWORDS)
    has_retrieval = any(r in skills_set for r in RETRIEVAL_KEYWORDS)
    has_ranking = any(rk in skills_set for rk in RANKING_KEYWORDS)
    has_eval = any(e in skills_set for e in EVAL_KEYWORDS)
    has_lora = any(l in skills_set for l in LORA_KEYWORDS)
    
    # Base skills score
    skills_score = sum([has_vec_db, has_prod, has_retrieval, has_ranking, has_eval, has_lora]) / 6.0
    
    # 2. Career History Analysis (finding matching roles and text)
    matched_months = 0
    total_relevance_points = 0
    
    for job in history:
        title = job.get("title", "").lower()
        desc = job.get("description", "").lower()
        duration = job.get("duration_months", 0)
        
        # Check if the title is relevant to ML/AI/Search
        is_title_relevant = any(w in title for w in TITLE_WEIGHTED_WORDS)
        weight = 2.0 if is_title_relevant else 1.0
        
        # Find term occurrences in description
        points = 0
        
        # Count unique vector DB matches
        db_matches = sum(1 for db in VEC_DBS if db in desc)
        points += min(db_matches, 3) * 1.5
        
        # Count production keywords
        prod_matches = sum(1 for kw in PROD_KEYWORDS if re.search(r'\b' + re.escape(kw) + r'\b', desc))
        points += min(prod_matches, 4) * 1.0
        
        # Count retrieval, ranking, eval, and lora keywords
        ret_matches = sum(1 for kw in RETRIEVAL_KEYWORDS if kw in desc)
        points += min(ret_matches, 3) * 1.5
        
        rank_matches = sum(1 for kw in RANKING_KEYWORDS if kw in desc)
        points += min(rank_matches, 3) * 1.5
        
        eval_matches = sum(1 for kw in EVAL_KEYWORDS if kw in desc)
        points += min(eval_matches, 3) * 1.5
        
        lora_matches = sum(1 for kw in LORA_KEYWORDS if kw in desc)
        points += min(lora_matches, 2) * 1.0
        
        if points > 1.5:
            # This role has actual production ML content
            total_relevance_points += points * weight
            matched_months += duration
            
    # Normalize matched months: 36 months (3 years) of production experience is a solid baseline
    duration_factor = min(matched_months / 48.0, 1.0) # Cap at 4 years
    
    # Combine skills score and description points
    desc_factor = min(total_relevance_points / 25.0, 1.0)
    
    production_score = 0.3 * skills_score + 0.4 * desc_factor + 0.3 * duration_factor
    
    return min(max(production_score, 0.0), 1.0)
