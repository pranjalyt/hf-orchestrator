# agents/quality_agent.py
# Evaluates model candidates and injects chaos (503 errors).
# This is what tests the orchestrator's RESILIENCE.

import random
from typing import List


def evaluate(
    model_ids: List[str],
    task_context: str,
    has_enterprise_license: bool = False,
    chaos_enabled: bool = True
) -> dict:
    """
    Evaluates a list of candidate models.
    
    THE KEY MECHANIC: 10% chance of returning 503.
    This forces the orchestrator to learn wait_and_retry behavior.
    
    Args:
        model_ids: List of model IDs to evaluate
        task_context: What the models need to do
        has_enterprise_license: Whether gated models are accessible
        chaos_enabled: Toggle chaos for Phase 1 training (happy path)
    
    Returns:
        Either evaluation results OR error response
    """
    
    # CHAOS INJECTION - 10% failure rate
    # Simulates real enterprise API flakiness
    if chaos_enabled and random.random() < 0.10:
        return {
            "status": "error",
            "error_code": 503,
            "message": "Service Unavailable. Quality service is temporarily down.",
            "retry_after": 2  # Hint: wait 2 steps then retry
        }
    
    evaluations = []
    
    for model_id in model_ids:
        evaluation = _evaluate_single_model(
            model_id, 
            task_context,
            has_enterprise_license
        )
        evaluations.append(evaluation)
    
    # Sort by quality score descending
    evaluations.sort(key=lambda x: x["quality_score"], reverse=True)
    
    return {
        "status": "success",
        "evaluations": evaluations,
        "top_pick": evaluations[0]["model_id"] if evaluations else None,
        "recommendation": evaluations[0]["recommendation"] if evaluations else None
    }


def _evaluate_single_model(
    model_id: str,
    task_context: str,
    has_enterprise_license: bool
) -> dict:
    """
    Scores a single model on multiple quality dimensions.
    Uses real metadata from our database where available.
    """
    # from scripts.scrape_hf_hub import DATABASE  # Use real metadata
    from agents.search_agent import DATABASE
    
    # Find model in database
    model_data = _find_model_in_db(model_id)
    
    # Check gated access
    if model_data.get("is_gated") and not has_enterprise_license:
        return {
            "model_id": model_id,
            "quality_score": 0.0,
            "recommendation": "BLOCKED",
            "reason": "401 Access Denied. Model is gated. Enterprise license required.",
            "error_code": 401
        }
    
    # Calculate quality score from multiple signals
    download_score = min(model_data.get("downloads", 0) / 1_000_000, 1.0)
    
    # Recency score (newer = better maintained)
    recency_score = _calculate_recency_score(
        model_data.get("last_modified", "2020-01-01")
    )
    
    # Task alignment score (does model match task?)
    alignment_score = _calculate_alignment(
        model_data.get("task", ""), 
        task_context
    )
    
    # Weighted composite score
    quality_score = (
        download_score * 0.4 +
        recency_score * 0.3 +
        alignment_score * 0.3
    )
    
    # Determine recommendation
    if quality_score >= 0.7:
        recommendation = "STRONG_YES"
    elif quality_score >= 0.4:
        recommendation = "YES"
    elif quality_score >= 0.2:
        recommendation = "MAYBE"
    else:
        recommendation = "NO"
    
    return {
        "model_id": model_id,
        "quality_score": round(quality_score, 3),
        "recommendation": recommendation,
        "breakdown": {
            "download_score": round(download_score, 3),
            "recency_score": round(recency_score, 3),
            "alignment_score": round(alignment_score, 3)
        },
        "estimated_vram_gb": model_data.get("estimated_vram_gb", 2.0),
        "is_gated": model_data.get("is_gated", False)
    }


def _find_model_in_db(model_id: str) -> dict:
    """Finds a model in the local database by ID."""
    from agents.search_agent import DATABASE
    
    for task_models in DATABASE.values():
        for model in task_models:
            if model["id"] == model_id:
                return model
    
    # Model not in DB - return minimal defaults
    return {"downloads": 0, "last_modified": "2020-01-01", "task": "unknown"}


def _calculate_recency_score(last_modified: str) -> float:
    """
    Scores how recently maintained a model is.
    2024 = 1.0, 2020 = 0.0
    """
    try:
        year = int(str(last_modified)[:4])
        return min(max((year - 2019) / 5.0, 0.0), 1.0)
    except:
        return 0.3  # Default if parsing fails


def _calculate_alignment(model_task: str, task_context: str) -> float:
    """
    Simple keyword matching to check if model fits the task.
    """
    task_context_lower = task_context.lower()
    model_task_lower = model_task.lower()
    
    # Direct match
    if model_task_lower in task_context_lower:
        return 1.0
    
    # Partial matches
    keyword_map = {
        "text-classification": ["classify", "classification", "sentiment", "toxic"],
        "text-generation": ["generate", "generation", "completion", "chat"],
        "summarization": ["summarize", "summary", "tldr"],
        "question-answering": ["qa", "question", "answer"],
        "token-classification": ["ner", "entity", "token", "label"]
    }
    
    for task, keywords in keyword_map.items():
        if task == model_task_lower:
            if any(kw in task_context_lower for kw in keywords):
                return 0.8
    
    return 0.2  # Low but not zero - might still work