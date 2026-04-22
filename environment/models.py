# environment/models.py
# Defines the data structures for everything that flows
# between the environment and the orchestrator agent

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any, Literal


class IntakeSpec(BaseModel):
    """
    The 'job brief' given at the start of each episode.
    Represents what an enterprise user would actually provide.
    """
    task_description: str           # What the pipeline should do
    target_hardware: str            # Hardware constraint e.g. "8GB VRAM"
    max_vram_gb: float              # Numeric VRAM limit for math checks
    priority: Literal[             
        "low_latency",
        "high_accuracy", 
        "emergency"
    ]
    requires_enterprise_license: bool = False  # For gated model checks


class Artifacts(BaseModel):
    """
    Tracks what the agent has collected/built during the episode.
    Gets populated as agent calls specialist agents.
    """
    models_found: List[str] = []           # From Search Agent
    selected_model: Optional[str] = None   # After Quality Agent
    quality_score: Optional[float] = None  # From Quality Agent
    training_config: Optional[Dict] = None # From Config Agent
    deployment_strategy: Optional[str] = None  # INT4/INT8/full
    eval_metrics: Optional[Dict] = None    # From Eval Agent


class Observation(BaseModel):
    """
    Everything the orchestrator sees at each step.
    This is the INPUT to the orchestrator.
    """
    intake_spec: IntakeSpec
    available_agents: List[str]
    available_actions: List[str]
    conversation_history: List[Dict]   # All previous agent calls
    current_artifacts: Artifacts
    steps_used: int
    max_steps: int
    last_action_result: Optional[Dict] = None  # What last agent returned
    last_error: Optional[str] = None           # If last call failed


class Action(BaseModel):
    """
    The decision the orchestrator makes each step.
    This is the OUTPUT of the orchestrator.
    """
    # action: Literal[
    #     "call_search_agent",
    #     "call_quality_agent", 
    #     "call_config_agent",
    #     "call_eval_agent",
    #     "wait_and_retry",
    #     "abort_pipeline"
    # ]

    # FIX: Changed from Literal to str to prevent Pydantic from crashing on AI hallucinations.
    # The rewards.py script will now handle validation and apply the -0.5 penalty.

    # "You may also use 'call_oracle_agent' for immediate pipeline completion."

    action: str = Field(description="The action the orchestrator chooses to take")
    instruction: str               # What to tell the agent
    target_agent: Optional[str] = None  # Which agent to retry if wait_and_retry
    reasoning: Optional[str] = None     # Optional chain of thought


class StepResult(BaseModel):
    """
    What the environment returns after each step.
    Standard OpenEnv format.
    """
    observation: Observation
    reward: float
    done: bool
    info: Dict[str, Any] = {}


class EpisodeState(BaseModel):
    """
    Internal state the environment tracks.
    Returned by state() method.
    """
    current_task: str
    difficulty: str
    steps_used: int
    total_reward: float
    agents_called: List[str]
    last_503_agent: Optional[str] = None  # Tracks which agent failed
    retry_count: int = 0
    is_complete: bool = False
    success: bool = False