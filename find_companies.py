import json
from pathlib import Path
from collections import Counter

DATA_DIR = Path("C:/Users/Gopika Arasi/OneDrive/Documents/Desktop/indiarun/[PUB] India_runs_data_and_ai_challenge/[PUB] India_runs_data_and_ai_challenge/India_runs_data_and_ai_challenge")
CANDIDATES_FILE = DATA_DIR / "candidates.jsonl"

def find_unique_companies():
    all_companies = Counter()
    with open(CANDIDATES_FILE, "r", encoding="utf-8") as f:
        for line in f:
            cand = json.loads(line)
            for job in cand.get("career_history", []):
                all_companies[job.get("company", "").strip()] += 1
                
    # Print companies sorted by name
    sorted_comps = sorted(list(all_companies.keys()))
    print(f"Total unique companies: {len(sorted_comps)}")
    print("\n--- SAMPLE COMPANIES ---")
    for c in sorted_comps[:100]:
        print(c)
        
    print("\n--- COMPANIES CONTAINING AI OR VECTOR OR CORE TECH NAMES ---")
    tech_keywords = ["openai", "anthropic", "cohere", "pinecone", "weaviate", "qdrant", "milvus", "chroma", "faiss", "hugging", "google", "meta", "microsoft", "amazon", "netflix", "apple"]
    for c in sorted_comps:
        for k in tech_keywords:
            if k in c.lower():
                print(f"{c}: {all_companies[c]} times")
                break

find_unique_companies()
