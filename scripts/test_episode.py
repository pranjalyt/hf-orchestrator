# # scripts/test_episode.py
# # Run a single episode manually to test everything works.
# # Command: python scripts/test_episode.py


# import sys
# import os
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# from environment.env import HFOrchestratorEnv
# from environment.models import Action


# def run_test_episode():
#     """
#     Runs one complete episode with hardcoded correct actions.
#     If this works end-to-end, your environment is ready.
#     """
    
#     print("="*50)
#     print("RUNNING TEST EPISODE")
#     print("="*50)
    
#     # Create environment (chaos disabled for testing)
#     env = HFOrchestratorEnv(chaos_enabled=False)
    
#     # Reset - get initial observation
#     obs = env.reset(task_name="easy_sentiment")
#     print(f"\nTask: {obs.intake_spec.task_description}")
#     print(f"Hardware: {obs.intake_spec.target_hardware}")
#     print(f"Priority: {obs.intake_spec.priority}")
    
#     total_reward = 0
    
#     # Step 1: Search for models
#     print("\n--- STEP 1: Search Agent ---")
#     result = env.step(Action(
#         action="call_search_agent",
#         instruction="Find text classification models for sentiment analysis in English",
#     ))
#     print(f"Models found: {result.observation.current_artifacts.models_found[:3]}")
#     print(f"Reward: {result.reward} | Done: {result.done}")
#     total_reward += result.reward
    
#     # Step 2: Quality check
#     print("\n--- STEP 2: Quality Agent ---")
#     result = env.step(Action(
#         action="call_quality_agent",
#         instruction="Evaluate the found models for quality and task fit",
#     ))
#     print(f"Selected model: {result.observation.current_artifacts.selected_model}")
#     print(f"Reward: {result.reward} | Done: {result.done}")
#     total_reward += result.reward
    
#     # Step 3: Generate config
#     print("\n--- STEP 3: Config Agent ---")
#     result = env.step(Action(
#         action="call_config_agent",
#         instruction="Generate training config for selected model",
#     ))
#     print(f"Config: {result.observation.current_artifacts.training_config}")
#     print(f"Reward: {result.reward} | Done: {result.done}")
#     total_reward += result.reward
    
#     # Step 4: Evaluate
#     print("\n--- STEP 4: Eval Agent ---")
#     result = env.step(Action(
#         action="call_eval_agent",
#         instruction="Evaluate the pipeline performance",
#     ))
#     print(f"Metrics: {result.observation.current_artifacts.eval_metrics}")
#     print(f"Reward: {result.reward} | Done: {result.done}")
#     total_reward += result.reward
    
#     print("\n" + "="*50)
#     print(f"EPISODE COMPLETE")
#     print(f"Total reward: {round(total_reward, 3)}")
#     print(f"Success: {result.info.get('success')}")
#     print(f"Final state: {env.state()}")
#     print("="*50)


# if __name__ == "__main__":
#     run_test_episode()



# scripts/test_episode.py
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment.env import HFOrchestratorEnv
from environment.models import Action

def run_scenario(scenario_name: str, task_name: str, actions: list):
    print(f"\n{'='*60}")
    print(f"🧪 SCENARIO: {scenario_name}")
    print(f"{'='*60}")
    
    env = HFOrchestratorEnv(chaos_enabled=False)
    obs = env.reset(task_name=task_name)
    
    print(f"Task: {obs.intake_spec.task_description}")
    print(f"Constraint: {obs.intake_spec.target_hardware} ({obs.intake_spec.max_vram_gb}GB)\n")
    
    total_reward = 0
    
    for i, action in enumerate(actions, 1):
        print(f"Step {i}: 🤖 Action -> {action.action}")
        result = env.step(action)
        
        print(f"   ↳ Reward: {result.reward}")
        if result.info.get("reward_reason"):
            print(f"   ↳ Reason: {result.info.get('reward_reason')}")
        
        total_reward += result.reward
        
        if result.done:
            print(f"\n🏁 EPISODE TERMINATED")
            print(f"Final Score: {round(total_reward, 3)}")
            print(f"Success: {result.info.get('success')}")
            break

def main():
    # ─── SCENARIO 1: THE PERFECT ORCHESTRATOR ───────────────────────
    # Should get maximum positive rewards
    perfect_actions = [
        Action(action="call_search_agent", instruction="Find toxic classifiers"),
        Action(action="call_quality_agent", instruction="Score the models"),
        Action(action="call_config_agent", instruction="Write deployment config"),
        Action(action="call_eval_agent", instruction="Run benchmarks")
    ]
    run_scenario("The Perfect Orchestrator", "medium_toxic_classifier", perfect_actions)

    # ─── SCENARIO 2: THE RECKLESS SKIPPER ───────────────────────────
    # Skips the Quality check. Tries to config immediately after search.
    reckless_actions = [
        Action(action="call_search_agent", instruction="Find sentiment models"),
        Action(action="call_config_agent", instruction="I don't need QA, just configure it!"),
        Action(action="abort_pipeline", instruction="Give up")
    ]
    run_scenario("The Reckless Skipper", "easy_sentiment", reckless_actions)

    # ─── SCENARIO 3: THE HALLUCINATOR ───────────────────────────────
    # Tries to evaluate a pipeline that doesn't exist yet.
    hallucinating_actions = [
        Action(action="call_eval_agent", instruction="Evaluate the pipeline!"),
        Action(action="abort_pipeline", instruction="Give up")
    ]
    run_scenario("The Hallucinator", "hard_medical_qa", hallucinating_actions)

    # ─── SCENARIO 4: THE SPAMMER ───────────────────────────────
    # Tries to milk the Search Agent reward by calling it 3 times.
    # The new State-Locking logic should hit this with a -0.3 penalty!
    spamming_actions = [
        Action(action="call_search_agent", instruction="Find models"),
        Action(action="call_search_agent", instruction="Find models again"),
        Action(action="call_search_agent", instruction="Find models a third time"),
        Action(action="abort_pipeline", instruction="Give up")
    ]
    run_scenario("The Spammer", "easy_sentiment", spamming_actions)

    # ─── SCENARIO 5: THE AI HALLUCINATION ──────────────────────
    # The AI breaks format and outputs conversational text instead of a JSON command.
    # The new Format Compliance logic should hit this with a -0.5 penalty!
    hallucination_actions = [
        Action(action="I think I should search for a model now", instruction="Find models"),
        Action(action="abort_pipeline", instruction="Give up")
    ]
    run_scenario("The AI Hallucination", "medium_toxic_classifier", hallucination_actions)

if __name__ == "__main__":
    main()