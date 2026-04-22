# agents/eval_agent.py
# Simulates benchmark evaluation using parameter-math heuristics.
# No actual model inference happens here - just math.

def evaluate_pipeline(
    model_id: str,
    task_type: str,
    training_config: dict,
    deployment_blueprint: dict,
    hardware_constant: float = 10.0
) -> dict:
    """
    Simulates what a trained model's performance would be.
    
    Uses heuristics based on:
    - Model size (bigger = more accurate, slower)
    - Quantization (reduces both latency AND accuracy slightly)
    - Training config quality
    - Task difficulty
    
    Args:
        model_id: The trained model
        task_type: What task we're evaluating
        training_config: Config from Config Agent
        deployment_blueprint: Deployment from Config Agent
        hardware_constant: Latency multiplier per billion params
    
    Returns:
        Simulated metrics + deployment recommendation
    """
    
    from agents.quality_agent import _find_model_in_db
    model_data = _find_model_in_db(model_id)
    
    params_b = model_data.get("estimated_params_b", 0.11)
    quant = deployment_blueprint.get("serving_framework", "FULL_FP16")
    
    # Simulate accuracy based on model size
    # Bigger models = higher accuracy ceiling
    base_accuracy = _simulate_accuracy(params_b, task_type)
    
    # Quantization penalty
    accuracy = _apply_quantization_penalty(base_accuracy, quant)
    
    # Simulate latency
    latency_ms = _simulate_latency(params_b, quant, hardware_constant)

    # Simulate cost
    cost = _simulate_cost(params_b, quant)
    
    # Check if latency meets target
    target_ms = deployment_blueprint.get("latency_target_ms", 200)
    latency_ok = latency_ms <= target_ms
    
    # Bias check (simplified)
    bias_score = _simulate_bias_score(model_id, task_type)
    
    # Safety check
    safety_ok = bias_score < 0.3
    
    # Final recommendation
    if accuracy >= 0.85 and latency_ok and safety_ok:
        recommendation = "DEPLOY"
    elif accuracy >= 0.75 and safety_ok:
        recommendation = "DEPLOY_WITH_MONITORING"
    else:
        recommendation = "RETRAIN"
    
    return {
        "status": "success",
        "metrics": {
            "simulated_accuracy": round(accuracy, 3),
            "simulated_f1": round(accuracy * 0.97, 3),
            "latency_ms": round(latency_ms, 1),
            "latency_target_met": latency_ok,
            "bias_score": round(bias_score, 3),
            "safety_threshold_met": safety_ok,
            "estimated_cost_per_1m": round(cost, 4)
        },
        "recommendation": recommendation,
        "model_card_notes": _generate_model_card_notes(
            accuracy, latency_ms, bias_score
        )
    }

def _simulate_cost(params_b: float, framework: str) -> float:
    """Simulates cost per 1M tokens based on model size and quantization."""
    base_cost = params_b * 0.50 # 50 cents per billion params
    if "llama.cpp" in framework: # INT4 saves massive compute costs
        return base_cost * 0.25
    return base_cost

def _simulate_accuracy(params_b: float, task_type: str) -> float:
    """
    Bigger models are more accurate.
    Accuracy = 0.65 + log_scale(params) capped at 0.97.
    """
    import math
    
    # Log scale: 0.1B=0.70, 1B=0.80, 7B=0.88, 70B=0.96
    base = 0.65 + (math.log10(params_b + 0.1) + 1) * 0.12
    base = min(base, 0.97)
    
    # Classification is easier than generation
    if "classification" in task_type:
        base = min(base + 0.03, 0.97)
    
    return base


def _apply_quantization_penalty(accuracy: float, framework: str) -> float:
    """INT4 costs ~3% accuracy, INT8 costs ~1%."""
    
    if "llama.cpp" in framework:  # INT4
        return accuracy * 0.97
    elif "bitsandbytes" in framework:  # INT8
        return accuracy * 0.99
    else:  # Full precision
        return accuracy


def _simulate_latency(
    params_b: float,
    framework: str,
    hardware_constant: float
) -> float:
    """
    Base latency = (params_b / 1B) * hardware_constant ms.
    INT8 halves latency, INT4 quarters it.
    """
    
    base_latency = params_b * hardware_constant
    
    if "llama.cpp" in framework:  # INT4 = 4x faster
        return base_latency / 4
    elif "bitsandbytes" in framework:  # INT8 = 2x faster
        return base_latency / 2
    else:
        return base_latency


def _simulate_bias_score(model_id: str, task_type: str) -> float:
    """
    Random bias score weighted by task sensitivity.
    Higher for classification tasks (more prone to bias).
    """
    import random
    
    if "classification" in task_type:
        return random.uniform(0.05, 0.35)
    return random.uniform(0.02, 0.20)


def _generate_model_card_notes(
    accuracy: float,
    latency_ms: float,
    bias_score: float
) -> str:
    """Generates model card text for documentation."""
    
    notes = []
    
    if accuracy >= 0.90:
        notes.append("High accuracy model suitable for production use.")
    elif accuracy >= 0.80:
        notes.append("Good accuracy. Monitor for edge cases.")
    else:
        notes.append("Moderate accuracy. Consider larger model.")
    
    if latency_ms < 50:
        notes.append("Excellent latency for real-time applications.")
    elif latency_ms < 200:
        notes.append("Acceptable latency for most use cases.")
    else:
        notes.append("High latency. Consider quantization or smaller model.")
    
    if bias_score > 0.2:
        notes.append("WARNING: Elevated bias detected. Review fairness metrics.")
    
    return " ".join(notes)