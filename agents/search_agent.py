# agents/search_agent.py
# Searches the mock HF Hub database for relevant models.
# Part of the environment - NOT what we train.

import json
import random
from pathlib import Path


def load_database() -> dict:
    """
    Loads the mock HF Hub JSON from disk.
    Called once when agent is first used.
    """
    db_path = Path("data/hf_hub_snapshot.json")
    
    if not db_path.exists():
        raise FileNotFoundError(
            "Database not found. Run: python scripts/scrape_hf_hub.py"
        )
    
    with open(db_path) as f:
        return json.load(f)


# Load database once at module level (not on every call)
DATABASE = load_database()


def search(
    instruction: str,
    task_type: str = None,
    max_vram_gb: float = None,
    top_k: int = 5
) -> dict:
    """
    Main search function called by the environment.
    
    Args:
        instruction: Natural language search query from orchestrator
        task_type: Optional filter (text-classification, etc.)
        max_vram_gb: Optional hardware constraint filter
        top_k: How many results to return
    
    Returns:
        Dict with results list and metadata
    
    The shuffle is INTENTIONAL - prevents agent from learning
    lazy habit of always picking index 0.
    """
    
    # Determine which tasks to search
    if task_type and task_type in DATABASE:
        # Search specific task
        candidate_pool = DATABASE[task_type]
    else:
        # Search all tasks, flatten into one list
        candidate_pool = []
        for task_models in DATABASE.values():
            candidate_pool.extend(task_models)
    
    # Filter by hardware constraint if provided
    if max_vram_gb is not None:
        candidate_pool = [
            m for m in candidate_pool 
            if m.get("estimated_vram_gb", 999) <= max_vram_gb
        ]
    
    # Sort by downloads (quality signal)
    candidate_pool.sort(key=lambda x: x.get("downloads", 0), reverse=True)
    
    # Take top results
    top_results = candidate_pool[:top_k * 2]  # Get 2x then shuffle
    
    # INTENTIONAL SHUFFLE - forces agent to use Quality Agent
    # instead of lazy index-0 picking
    random.shuffle(top_results)
    
    # Return top_k after shuffle
    final_results = top_results[:top_k]
    
    return {
        "status": "success",
        "query": instruction,
        "total_found": len(candidate_pool),
        "results": final_results,
        "note": "Results are shuffled. Use Quality Agent to rank."
    }