# server.py
# FastAPI server that exposes the environment via HTTP.
# This is what HuggingFace Spaces runs.
# Judges ping /reset to validate your submission.
#
# Run locally: uvicorn server:app --host 0.0.0.0 --port 7860 --reload
# Docker runs this automatically via CMD in Dockerfile

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import uvicorn

from environment.env import HFOrchestratorEnv
from environment.models import Action

# ─── App Setup ────────────────────────────────────────────────

app = FastAPI(
    title="HF Hub Orchestrator",
    description="RL environment for ML pipeline orchestration",
    version="1.0.0"
)

# Allow all origins (needed for HF Spaces)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Global Environment Instance ──────────────────────────────
# One environment per server instance.
# In production you'd have one per session, but this
# is fine for hackathon/evaluation purposes.

env = HFOrchestratorEnv(chaos_enabled=True)
current_observation = None


# ─── Request/Response Models ───────────────────────────────────

class ResetRequest(BaseModel):
    """Optional: specify which task to load."""
    task_name: Optional[str] = None


class StepRequest(BaseModel):
    """Action from the agent."""
    action: str
    instruction: str
    target_agent: Optional[str] = None
    reasoning: Optional[str] = None


# ─── Endpoints ────────────────────────────────────────────────

@app.get("/health")
def health_check():
    """
    Basic health check.
    Returns 200 if server is running.
    Judges use this to verify deployment.
    """
    return {
        "status": "healthy",
        "environment": "hf-hub-orchestrator",
        "version": "1.0.0"
    }


@app.get("/tasks")
def list_tasks():
    """
    Returns all available tasks.
    Judges use this to enumerate tasks for grading.
    """
    return {
        "tasks": [
            {
                "name": "easy_sentiment",
                "difficulty": "easy",
                "description": "Find pre-trained English sentiment classifier"
            },
            {
                "name": "medium_toxic_classifier", 
                "difficulty": "medium",
                "description": "Build toxic content classifier under 8GB VRAM constraint"
            },
            {
                "name": "hard_medical_qa",
                "difficulty": "hard",
                "description": "Build production medical QA under 4GB VRAM with API chaos"
            }
        ]
    }


@app.post("/reset")
def reset(request: ResetRequest = None):
    """
    Starts a fresh episode.
    
    This is the FIRST endpoint judges ping.
    Must return 200 or you're disqualified.
    
    Returns initial observation with intake spec.
    """
    global current_observation
    
    try:
        task_name = request.task_name if request else None
        current_observation = env.reset(task_name=task_name)
        
        return {
            "status": "success",
            "observation": current_observation.dict()
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/step")
def step(request: StepRequest):
    """
    Agent submits action, environment responds.
    
    Returns new observation + reward + done flag.
    Judges run their agent against this endpoint.
    """
    global current_observation
    
    if current_observation is None:
        raise HTTPException(
            status_code=400,
            detail="Environment not initialized. Call /reset first."
        )
    
    try:
        # Build action from request
        action = Action(
            action=request.action,
            instruction=request.instruction,
            target_agent=request.target_agent,
            reasoning=request.reasoning
        )
        
        # Step environment
        result = env.step(action)
        current_observation = result.observation
        
        return {
            "status": "success",
            "observation": result.observation.dict(),
            "reward": result.reward,
            "done": result.done,
            "info": result.info
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/state")
def get_state():
    """
    Returns current internal state.
    Used for debugging and monitoring during evaluation.
    """
    return {
        "status": "success",
        "state": env.state()
    }


# ─── Entry Point ──────────────────────────────────────────────

if __name__ == "__main__":
    # HF Spaces expects port 7860
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=7860,
        reload=False  # No reload in production
    )