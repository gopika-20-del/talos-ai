import json
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("TalosAI")

def find_data_dir():
    """
    Finds the data directory containing candidates.jsonl recursively in the workspace.
    """
    check_paths = [
        Path("./[PUB] India_runs_data_and_ai_challenge/[PUB] India_runs_data_and_ai_challenge/India_runs_data_and_ai_challenge"),
        Path("./[PUB] India_runs_data_and_ai_challenge"),
        Path("."),
    ]
    
    for p in check_paths:
        if (p / "candidates.jsonl").exists():
            return p.resolve()
            
    # Recursive search
    for p in Path(".").rglob("candidates.jsonl"):
        return p.parent.resolve()
        
    # Check parent workspace
    parent_workspace = Path("C:/Users/Gopika Arasi/OneDrive/Documents/Desktop/indiarun")
    for p in parent_workspace.rglob("candidates.jsonl"):
        return p.parent.resolve()
        
    raise FileNotFoundError("Could not locate candidates.jsonl in workspace.")

def load_candidates(limit=None):
    """
    Memory-efficient generator that yields candidate profiles one by one.
    """
    data_dir = find_data_dir()
    candidates_file = data_dir / "candidates.jsonl"
    
    logger.info(f"Loading candidates from {candidates_file}")
    
    count = 0
    with open(candidates_file, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            yield json.loads(line)
            count += 1
            if limit and count >= limit:
                break

def get_candidates_count():
    """
    Returns the total number of candidates.
    """
    data_dir = find_data_dir()
    candidates_file = data_dir / "candidates.jsonl"
    count = 0
    with open(candidates_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                count += 1
    return count
