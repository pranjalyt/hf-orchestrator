# agents/config_agent.py
# Generates training configs with hardware-aware quantization.
# The DEPLOYMENT INTELLIGENCE of the pipeline.

def generate_config(
    model_id: str,
    task_type: str,
    max_vram_gb: float,
    priority: str,
    dataset_size: int = 10000
) -> dict:
    """
    Generates a training + deployment blueprint.
    
    THE KEY MECHANIC: If model is too big for hardware,
    automatically prescribes quantization strategy.
    If model CANNOT fit even with quantization, returns fatal error.
    
    Args:
        model_id: The selected model
        task_type: What the model needs to do
        max_vram_gb: Hardware VRAM limit from intake spec
        priority: low_latency / high_accuracy / emergency
        dataset_size: Estimated training samples
    
    Returns:
        Full deployment blueprint or fatal error
    """
    
    # Get model's VRAM requirement
    model_vram = _get_model_vram(model_id)
    model_params = _get_model_params(model_id)
    
    # Determine quantization strategy based on hardware
    quant_strategy, actual_vram = _determine_quantization(
        model_vram, 
        max_vram_gb,
        priority
    )
    
    # If even INT4 doesn't fit - FATAL ERROR
    if quant_strategy == "IMPOSSIBLE":
        return {
            "status": "fatal_error",
            "message": f"Model {model_id} ({model_params}B params) cannot fit "
                      f"in {max_vram_gb}GB VRAM even with INT4 quantization. "
                      f"Return to Search Agent and find smaller model.",
            "required_vram_minimum": model_vram * 0.25,  # INT4 minimum
            "current_limit": max_vram_gb
        }
    
    # Generate hyperparameters based on model size + task
    hyperparams = _generate_hyperparams(
        model_params, 
        task_type,
        dataset_size,
        priority
    )
    
    # Generate deployment blueprint
    deployment = _generate_deployment_blueprint(
        model_id,
        quant_strategy,
        priority
    )
    
    return {
        "status": "success",
        "model_id": model_id,
        "quantization_strategy": quant_strategy,
        "actual_vram_required_gb": actual_vram,
        "fits_hardware": True,
        "training_config": hyperparams,
        "deployment_blueprint": deployment,
        "warning": _generate_warnings(quant_strategy, priority)
    }


def _get_model_vram(model_id: str) -> float:
    """Gets estimated VRAM from database."""
    from agents.quality_agent import _find_model_in_db
    model_data = _find_model_in_db(model_id)
    return model_data.get("estimated_vram_gb", 2.0)


def _get_model_params(model_id: str) -> float:
    """Gets estimated params from database."""
    from agents.quality_agent import _find_model_in_db
    # from scripts.scrape_hf_hub import estimate_params
    model_data = _find_model_in_db(model_id)
    return model_data.get("estimated_params_b", 0.11)


def _determine_quantization(
    model_vram: float,
    max_vram: float,
    priority: str
) -> tuple:
    """
    Decides quantization strategy.
    
    Returns (strategy, actual_vram_needed) tuple.
    
    Hierarchy:
    Full FP16 → INT8 → INT4 → IMPOSSIBLE
    """
    
    # Full precision fits
    if model_vram <= max_vram:
        return "FULL_FP16", model_vram
    
    # INT8 halves VRAM
    int8_vram = model_vram * 0.5
    if int8_vram <= max_vram:
        return "INT8", int8_vram
    
    # INT4 quarters VRAM
    int4_vram = model_vram * 0.25
    if int4_vram <= max_vram:
        return "INT4", int4_vram
    
    # Nothing fits
    return "IMPOSSIBLE", model_vram


def _generate_hyperparams(
    model_params: float,
    task_type: str,
    dataset_size: int,
    priority: str
) -> dict:
    """
    Generates training hyperparameters based on model size.
    Uses established best practices from literature.
    """
    
    # Learning rate: smaller models can handle higher LR
    if model_params < 1.0:
        lr = 5e-5
    elif model_params < 7.0:
        lr = 2e-5
    else:
        lr = 1e-5
    
    # Batch size: constrained by VRAM
    if model_params < 1.0:
        batch_size = 32
    elif model_params < 7.0:
        batch_size = 16
    else:
        batch_size = 4
    
    # Epochs: more data needs fewer epochs
    epochs = 5 if dataset_size < 5000 else 3
    
    # Max length: classification needs less context
    max_length = 128 if "classification" in task_type else 512
    
    return {
        "learning_rate": lr,
        "batch_size": batch_size,
        "num_epochs": epochs,
        "max_length": max_length,
        "warmup_ratio": 0.1,
        "scheduler": "cosine",
        "optimizer": "adamw"
    }


def _generate_deployment_blueprint(
    model_id: str,
    quant_strategy: str,
    priority: str
) -> dict:
    """
    Generates the serving/deployment configuration.
    """
    
    # Framework depends on quantization
    if quant_strategy == "INT4":
        framework = "llama.cpp"
        runtime = "GGUF format"
    elif quant_strategy == "INT8":
        framework = "bitsandbytes"
        runtime = "PyTorch + HF Transformers"
    else:
        framework = "HuggingFace Transformers"
        runtime = "PyTorch FP16"
    
    # Latency target depends on priority
    latency_target_ms = 50 if priority == "low_latency" else 200
    
    return {
        "serving_framework": framework,
        "runtime": runtime,
        "latency_target_ms": latency_target_ms,
        "recommended_hardware": "T4 GPU" if quant_strategy != "IMPOSSIBLE" else None,
        "docker_base_image": "pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime"
    }


def _generate_warnings(quant_strategy: str, priority: str) -> str:
    """Generates human-readable warnings about tradeoffs."""
    
    if quant_strategy == "INT4" and priority == "high_accuracy":
        return "WARNING: INT4 quantization may reduce accuracy by 2-5%. Consider larger hardware."
    elif quant_strategy == "INT8":
        return "INFO: INT8 quantization selected. Minimal accuracy loss expected."
    else:
        return None