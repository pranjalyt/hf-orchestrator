# environment/env.py
# The main OpenEnv environment class.
# This is what gets validated by openenv validate.

import json
import random
from pathlib import Path
from typing import Optional

from environment.models import (
    Observation, Action, StepResult, 
    EpisodeState, IntakeSpec, Artifacts
)
from environment.rewards import calculate_step_reward, calculate_terminal_reward
from agents.search_agent import search
from agents.quality_agent import evaluate
from agents.config_agent import generate_config
from agents.eval_agent import evaluate_pipeline


class HFOrchestratorEnv:
    """
    Main OpenEnv environment class.
    
    Simulates an enterprise ML pipeline orchestration challenge.
    The agent must navigate the HuggingFace ecosystem under
    hardware constraints and API chaos to build ML pipelines.
    """
    
    def __init__(self, chaos_enabled: bool = True):
        """
        Args:
            chaos_enabled: Toggle 503 errors.
                          False during Phase 1 (happy path) training.
                          True during Phase 3 training.
        """
        self.chaos_enabled = chaos_enabled
        self.max_steps = 15
        self._state = None
        self._tasks = self._load_tasks()
    
    def reset(self, task_name: Optional[str] = None) -> Observation:
        """
        Starts a fresh episode.
        Called at the start of every training episode.
        
        Args:
            task_name: Specific task to load. 
                      If None, picks random task.
        
        Returns:
            Initial observation with intake spec and empty artifacts.
        """
        
        # Pick task
        if task_name:
            task = self._get_task_by_name(task_name)
        else:
            task = random.choice(self._tasks)
        
        # Initialize fresh state
        self._state = EpisodeState(
            current_task=task["name"],
            difficulty=task["difficulty"],
            steps_used=0,
            total_reward=0.0,
            agents_called=[],
            last_503_agent=None,
            retry_count=0,
            is_complete=False,
            success=False
        )
        
        # Store task details for this episode
        self._current_task = task
        self._current_artifacts = Artifacts()
        self._last_action_result = None
        self._last_error = None
        
        # Build intake spec from task definition
        intake_spec = IntakeSpec(
            task_description=task["task_description"],
            target_hardware=task["target_hardware"],
            max_vram_gb=task["max_vram_gb"],
            priority=task["priority"],
            requires_enterprise_license=task.get("requires_license", False)
        )
        
        self._intake_spec = intake_spec
        ##
        self._conversation_history = []
        
        return Observation(
            intake_spec=intake_spec,
            available_agents=["search", "quality", "config", "eval"],
            available_actions=[
                "call_search_agent",
                "call_quality_agent",
                "call_config_agent", 
                "call_eval_agent",
                "wait_and_retry",
                "abort_pipeline"
            ],
            conversation_history=[],
            current_artifacts=self._current_artifacts,
            steps_used=0,
            max_steps=self.max_steps,
            last_action_result=None,
            last_error=None
        )
    
    def step(self, action: Action) -> StepResult:
        """
        Core step function. Agent takes action, environment responds.
        
        Flow:
        1. Validate action
        2. Route to correct specialist agent
        3. Update artifacts with agent result
        4. Calculate reward
        5. Check if episode is done
        6. Return new observation + reward + done flag
        
        Args:
            action: Orchestrator's decision
        
        Returns:
            StepResult with new observation, reward, done flag
        """
        
        if self._state is None:
            raise RuntimeError("Call reset() before step()")
        
        self._state.steps_used += 1

        # FIX 1: Capture the state BEFORE we execute the action
        state_for_reward = self._state.dict()
        
        # Route action to correct handler
        result = self._handle_action(action)
        
        # Calculate step reward
        step_reward, reward_reason = calculate_step_reward(
            action=action.action,
            action_result=result,
            current_state=state_for_reward,  # Use the state from BEFORE the action
            intake_spec=self._intake_spec.dict()
        )
        
        self._state.total_reward += step_reward
        # GRADIENT PROTECTION: Prevent runaway negative scores
        self._state.total_reward = max(self._state.total_reward, -2.5)
        
        # Check if done
        done, success = self._check_done(action, result)
        
        if done:
            # Calculate terminal reward
            terminal_reward, terminal_reason = calculate_terminal_reward(
                success=success,
                steps_used=self._state.steps_used,
                max_steps=self.max_steps,
                eval_metrics=self._current_artifacts.eval_metrics,
                agents_called=self._state.agents_called
            )
            step_reward += terminal_reward
            self._state.total_reward += terminal_reward
            self._state.success = success
        
        # Update conversation history
        history_entry = {
            "step": self._state.steps_used,
            "action": action.action,
            "instruction": action.instruction,
            "result_status": result.get("status", "unknown") if result else "none",
            "reward": step_reward,
            "reward_reason": reward_reason
        }
        
        # Build new observation
        new_obs = Observation(
            intake_spec=self._intake_spec,
            available_agents=["search", "quality", "config", "eval"],
            available_actions=[
                "call_search_agent",
                "call_quality_agent",
                "call_config_agent",
                "call_eval_agent", 
                "wait_and_retry",
                "abort_pipeline"
            ],
            conversation_history=self._conversation_history + [history_entry],
            current_artifacts=self._current_artifacts,
            steps_used=self._state.steps_used,
            max_steps=self.max_steps,
            last_action_result=result,
            last_error=self._last_error
        )
        
        self._conversation_history = new_obs.conversation_history
        
        return StepResult(
            observation=new_obs,
            reward=step_reward,
            done=done,
            info={
                "success": success,
                "reward_reason": reward_reason,
                "total_reward": self._state.total_reward,
                "agents_called": self._state.agents_called
            }
        )
    
    def state(self) -> dict:
        """
        Returns current episode state.
        Used for debugging and monitoring.
        """
        if self._state is None:
            return {"status": "not_started"}
        
        return self._state.dict()
    
    def _handle_action(self, action: Action) -> Optional[dict]:
        """
        Routes action to correct specialist agent.
        Updates artifacts based on result.
        """
        
        self._last_error = None
        
        if action.action == "call_search_agent":
            result = search(
                instruction=action.instruction,
                task_type=self._current_task.get("task_type"),
                max_vram_gb=self._intake_spec.max_vram_gb
            )
            if result.get("status") == "success":
                self._current_artifacts.models_found = [
                    m["id"] for m in result.get("results", [])
                ]
                self._state.agents_called.append("search")
            return result
        
        elif action.action == "call_quality_agent":
            result = evaluate(
                model_ids=self._current_artifacts.models_found,
                task_context=self._intake_spec.task_description,
                has_enterprise_license=self._intake_spec.requires_enterprise_license,
                chaos_enabled=self.chaos_enabled
            )
            if result.get("status") == "error":
                # 503 happened - track it
                self._state.last_503_agent = "quality"
                self._last_error = result.get("message")
            elif result.get("status") == "success":
                self._current_artifacts.selected_model = result.get("top_pick")
                self._current_artifacts.quality_score = (
                    result["evaluations"][0]["quality_score"] 
                    if result.get("evaluations") else None
                )
                self._state.agents_called.append("quality")
                self._state.last_503_agent = None  # Reset on success
            return result
        
        elif action.action == "call_config_agent":
            if not self._current_artifacts.selected_model:
                return {
                    "status": "error",
                    "message": "No model selected yet. Call Quality Agent first."
                }
            result = generate_config(
                model_id=self._current_artifacts.selected_model,
                task_type=self._current_task.get("task_type", "text-classification"),
                max_vram_gb=self._intake_spec.max_vram_gb,
                priority=self._intake_spec.priority
            )
            if result.get("status") == "success":
                self._current_artifacts.training_config = result.get("training_config")
                self._current_artifacts.deployment_strategy = result.get("quantization_strategy")
                self._state.agents_called.append("config")
            return result
        
        elif action.action == "call_eval_agent":
            if not self._current_artifacts.training_config:
                return {
                    "status": "error",
                    "message": "No training config yet. Call Config Agent first."
                }
            result = evaluate_pipeline(
                model_id=self._current_artifacts.selected_model,
                task_type=self._current_task.get("task_type", "text-classification"),
                training_config=self._current_artifacts.training_config,
                deployment_blueprint={"serving_framework": "HuggingFace Transformers"}
            )
            if result.get("status") == "success":
                self._current_artifacts.eval_metrics = result.get("metrics")
                self._state.agents_called.append("eval")
            return result
        
        elif action.action == "wait_and_retry":
            self._state.retry_count += 1
            return {
                "status": "waiting",
                "message": "Waiting before retry...",
                "retry_count": self._state.retry_count
            }
        
        elif action.action == "abort_pipeline":
            return {
                "status": "aborted",
                "message": "Pipeline aborted by orchestrator."
            }
        
        return None
    
    def _check_done(self, action: Action, result: Optional[dict]) -> tuple:
        """
        Determines if episode should end.
        
        Returns (done, success) tuple.
        """
        
        # Max steps reached
        if self._state.steps_used >= self.max_steps:
            return True, False
        
        # Agent chose to abort
        if action.action == "abort_pipeline":
            return True, False
        
        # Successfully completed full pipeline
        if (self._current_artifacts.eval_metrics and 
            result and 
            result.get("recommendation") in ["DEPLOY", "DEPLOY_WITH_MONITORING"]):
            return True, True
        
        return False, False
    
    def _load_tasks(self) -> list:
        """Loads task dataset from JSON file."""
        tasks_path = Path("data/task_dataset.json")
        
        if not tasks_path.exists():
            # Return minimal default tasks if file doesn't exist yet
            return self._get_default_tasks()
        
        with open(tasks_path) as f:
            return json.load(f)
    
    def _get_task_by_name(self, name: str) -> dict:
        """Gets specific task by name."""
        for task in self._tasks:
            if task["name"] == name:
                return task
        raise ValueError(f"Task '{name}' not found")
    
    def _get_default_tasks(self) -> list:
        """Fallback tasks if JSON not built yet."""
        return [
            {
                "name": "easy_sentiment",
                "difficulty": "easy",
                "task_description": "Find pre-trained English sentiment classifier",
                "task_type": "text-classification",
                "target_hardware": "Standard GPU 16GB VRAM",
                "max_vram_gb": 16.0,
                "priority": "high_accuracy"
            },
            {
                "name": "medium_toxic_classifier",
                "difficulty": "medium",
                "task_description": "Build toxic content classifier for social media",
                "task_type": "text-classification",
                "target_hardware": "Edge Device 8GB VRAM",
                "max_vram_gb": 8.0,
                "priority": "low_latency"
            },
            {
                "name": "hard_medical_qa",
                "difficulty": "hard",
                "task_description": "Build production medical QA system with safety eval",
                "task_type": "question-answering",
                "target_hardware": "Constrained Edge 4GB VRAM",
                "max_vram_gb": 4.0,
                "priority": "high_accuracy",
                "requires_license": True
            }
        ]