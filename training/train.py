# training/train.py

import os
import json
import wandb
import torch
from unsloth import FastLanguageModel
from trl import PPOConfig, PPOTrainer

# ─── CONFIGURATION ──────────────────────────────────────────
DEBUG_MODE = True # ⚠️ KEEP TRUE FOR YOUR FREE T4 SMOKE TEST!

if DEBUG_MODE:
    print("⚠️ DEBUG MODE ACTIVE: Running fast test...")
    PHASE_EPISODES = [2, 2, 2] # Tiny phases
    MAX_STEPS = 5
else:
    PHASE_EPISODES = [80, 70, 50] # Full 200-episode curriculum
    MAX_STEPS = 15

# ─── CORE FUNCTIONS ─────────────────────────────────────────

def setup_wandb():
    wandb.init(
        project="hf-hub-orchestrator",
        name="ppo-curriculum-training",
        config={
            "model": "Qwen/Qwen2.5-1.5B-Instruct",
            "algorithm": "PPO (Multi-step Gym)",
            "total_episodes": sum(PHASE_EPISODES)
        }
    )

def load_model():
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/Qwen2.5-1.5B-Instruct",
        max_seq_length=2048,
        load_in_4bit=True,
        dtype=None,
    )
    
    # CRITICAL: TRL requires a pad token for batching
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha=16,
        lora_dropout=0.0,
        bias="none",
        use_gradient_checkpointing=True,
    )
    return model, tokenizer

def build_prompt(observation: dict, tokenizer) -> str:
    """Formats the state into Qwen's native ChatML structure."""
    artifacts = observation["current_artifacts"]
    history = observation["conversation_history"]
    
    hist_text = "".join([f"Step {h['step']}: {h['action']} -> {h['result_status']}\n" for h in history[-3:]])
    
    raw_text = f"""Task: {observation['intake_spec']['task_description']}
Constraint: {observation['intake_spec']['max_vram_gb']}GB VRAM

Models Found: {artifacts['models_found']}
Config Written: {'Yes' if artifacts['training_config'] else 'No'}

History:
{hist_text if hist_text else 'None'}

Choose next action."""

    messages = [
        {"role": "system", "content": "You are an orchestration AI. Always output valid JSON with 'action' and 'instruction' keys."},
        {"role": "user", "content": raw_text}
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

def parse_action(model_output: str) -> dict:
    import re
    try:
        return json.loads(model_output.strip())
    except:
        json_match = re.search(r'\{.*\}', model_output, re.DOTALL)
        return json.loads(json_match.group()) if json_match else {"action": "abort_pipeline", "instruction": "Format Error"}

# ─── PPO TRAINING ENGINE ────────────────────────────────────

def run_episode(model, tokenizer, env, chaos_enabled=False):
    """Runs a single complete environment episode."""
    from environment.models import Action
    
    env.chaos_enabled = chaos_enabled
    obs = env.reset()
    episode_data = []
    total_reward = 0.0
    done = False
    
    while not done and env._state.steps_used < MAX_STEPS:
        prompt = build_prompt(obs.dict(), tokenizer)
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs, 
                max_new_tokens=128, 
                temperature=0.7, 
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        # Extract just the generated tokens
        resp_tensor = outputs[0][inputs['input_ids'].shape[1]:]
        response_text = tokenizer.decode(resp_tensor, skip_special_tokens=True)
        
        action_dict = parse_action(response_text)
        try:
            action = Action(action=action_dict.get("action", "abort_pipeline"), instruction=action_dict.get("instruction", ""))
        except:
            action = Action(action="abort_pipeline", instruction="Format Error")
            
        result = env.step(action)
        
        # Save Tensors exactly as PPO requires them
        episode_data.append({
            "query": inputs['input_ids'][0],      # 1D LongTensor
            "response": resp_tensor,              # 1D LongTensor
            "reward": torch.tensor(result.reward, dtype=torch.float) # 0D FloatTensor
        })
        
        total_reward += result.reward
        obs = result.observation
        done = result.done
        
    return episode_data, total_reward


def train_phase(model, tokenizer, trainer, env, phase_name, episodes, chaos_enabled, start_ep):
    print(f"\n{'='*40}\nSTARTING {phase_name} ({episodes} episodes)\n{'='*40}")
    
    step_buffer = [] # NEW: Buffer to hold steps to satisfy exact PyTorch math

    for ep in range(episodes):
        episode_data, total_reward = run_episode(model, tokenizer, env, chaos_enabled)
        step_buffer.extend(episode_data)
        
        # The Magic: Only train when we have collected EXACTLY the batch size
        while len(step_buffer) >= trainer.config.batch_size:
            # Slice off a mathematically perfect batch
            batch = step_buffer[:trainer.config.batch_size]
            step_buffer = step_buffer[trainer.config.batch_size:]
            
            queries = [step["query"] for step in batch]
            responses = [step["response"] for step in batch]
            rewards = [step["reward"] for step in batch]
            
            trainer.step(queries, responses, rewards)
            
        global_ep = start_ep + ep
        wandb.log({"episode": global_ep, "episode_reward": total_reward, "phase": phase_name})
        print(f"Phase {phase_name} | Episode {global_ep} | Reward: {total_reward:.2f}")

    os.makedirs(f"checkpoints/{phase_name}", exist_ok=True)
    model.save_pretrained(f"checkpoints/{phase_name}")
    tokenizer.save_pretrained(f"checkpoints/{phase_name}")


def main():
    setup_wandb()
    model, tokenizer = load_model()
    
    # FIXED MATH: batch_size(4) = mini_batch(1) * grad_acc(4)
    config = PPOConfig(
        learning_rate=1.41e-5,
        batch_size=4,
        mini_batch_size=1,
        gradient_accumulation_steps=4,
    )
    trainer = PPOTrainer(config=config, model=model, tokenizer=tokenizer)
    
    from environment.env import HFOrchestratorEnv
    env = HFOrchestratorEnv()
    
    train_phase(model, tokenizer, trainer, env, "Phase1_Happy", PHASE_EPISODES[0], False, 0)
    train_phase(model, tokenizer, trainer, env, "Phase2_Hard", PHASE_EPISODES[1], False, PHASE_EPISODES[0])
    train_phase(model, tokenizer, trainer, env, "Phase3_Chaos", PHASE_EPISODES[2], True, PHASE_EPISODES[0]+PHASE_EPISODES[1])
    
    print("\nTraining Complete! Saving final weights...")
    model.save_pretrained("checkpoints/final")
    tokenizer.save_pretrained("checkpoints/final")
    wandb.finish()

if __name__ == "__main__":
    main()