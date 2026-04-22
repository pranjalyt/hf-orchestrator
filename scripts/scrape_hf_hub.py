# scripts/scrape_hf_hub.py
# Run this ONE TIME to build the local mock database.
# Command: python scripts/scrape_hf_hub.py
# Output: data/hf_hub_snapshot.json

import json
import os
import time
from huggingface_hub import HfApi

def scrape_models(api: HfApi, task: str, limit: int = 100) -> list:
    """
    Fetches real model metadata from HF Hub for a given task.
    Returns list of model dicts with all fields we need.
    """
    models = []
    
    print(f"Scraping {limit} models for task: {task}")
    
    # Get models sorted by downloads (most popular first)
    results = api.list_models(
        filter=task,         # <-- Changed from task=task
        sort="downloads",
        # direction=-1,
        limit=limit
        # Removed cardData=True to prevent further compatibility errors
    )
    
    for model in results:
        try:
            model_dict = {
                "id": model.modelId,
                "task": task,
                "downloads": model.downloads or 0,
                "likes": model.likes or 0,
                "last_modified": model.lastModified.strftime("%Y-%m-%d") if model.lastModified else "2023-01-01",
                # "tags": model.tags or [],
                "tags": [t for t in (model.tags or []) if t in ["pytorch", "safetensors", "jax", "en", "onnx"]],
                "is_gated": "gated" in (model.tags or []),
                # Estimate params from model name (rough heuristic)
                "estimated_params_b": estimate_params(model.modelId),
                # Estimate VRAM needed
                "estimated_vram_gb": estimate_vram(model.modelId)
            }
            models.append(model_dict)
            
        except Exception as e:
            # Skip models with missing data
            print(f"Skipping {model.modelId}: {e}")
            continue
    
    return models


def estimate_params(model_id: str) -> float:
    """
    Rough estimate of parameter count from model name.
    Used by Config Agent for hardware constraint checks.
    Real production would use model card metadata.
    """
    model_id_lower = model_id.lower()
    
    # Check for size indicators in name
    if "70b" in model_id_lower: return 70.0
    if "34b" in model_id_lower: return 34.0
    if "13b" in model_id_lower: return 13.0
    if "7b" in model_id_lower:  return 7.0
    if "3b" in model_id_lower:  return 3.0
    if "1.5b" in model_id_lower: return 1.5
    if "1b" in model_id_lower:  return 1.0
    if "large" in model_id_lower: return 0.35
    if "base" in model_id_lower: return 0.11
    if "small" in model_id_lower: return 0.06
    if "tiny" in model_id_lower: return 0.03
    
    return 0.11  # Default: assume BERT-base size


def estimate_vram(model_id: str) -> float:
    """
    Estimates VRAM needed in GB based on parameter count.
    Rule of thumb: ~2GB per 1B params in FP16.
    """
    params = estimate_params(model_id)
    return params * 2.0


def main():
    """
    Main function. Scrapes 5 task types, 100 models each.
    Saves to data/hf_hub_snapshot.json
    """
    api = HfApi()
    
    # These 5 tasks cover most enterprise ML use cases
    tasks_to_scrape = [
        "text-classification",
        "text-generation", 
        "token-classification",
        "question-answering",
        "summarization"
    ]
    
    all_models = {}
    
    for task in tasks_to_scrape:
        models = scrape_models(api, task, limit=100)
        all_models[task] = models
        print(f"Got {len(models)} models for {task}")
        
        # Be nice to HF API - don't hammer it
        time.sleep(2)
    
    # Make sure data directory exists
    os.makedirs("data", exist_ok=True)
    
    # Save to JSON
    output_path = "data/hf_hub_snapshot.json"
    with open(output_path, "w") as f:
        json.dump(all_models, f, indent=2, default=str)
    
    # Count total
    total = sum(len(v) for v in all_models.values())
    print(f"\nDone! Saved {total} models to {output_path}")


if __name__ == "__main__":
    main()