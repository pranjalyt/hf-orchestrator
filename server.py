# server.py
from fastapi import FastAPI
from environment.env import HFOrchestratorEnv

# This is the 'app' variable Uvicorn is looking for
app = FastAPI(
    title="HF Orchestrator Environment",
    description="RL Training Server for OpenEnv"
)

# Boot up your custom environment in the background
env = HFOrchestratorEnv(chaos_enabled=False)

@app.get("/")
def health_check():
    """Hugging Face pings this to check if the Space is alive."""
    return {
        "status": "online", 
        "message": "Environment is running perfectly.",
        "architecture": "OpenEnv"
    }

@app.post("/reset")
def reset_environment():
    """Optional endpoint if you want to reset the env remotely."""
    obs = env.reset()
    return obs.dict()