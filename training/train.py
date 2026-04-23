# training/train.py

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import wandb
import torch
from typing import List

# 1. Standard, Enterprise-Grade Imports (No Unsloth)
from trl import PPOConfig, PPOTrainer, AutoModelForCausalLMWithValueHead
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# ─── CONFIGURATION ──────────────────────────────────────────
DEBUG_MODE = True # KEEP TRUE FOR YOUR FIRST COLAB TEST

if DEBUG_MODE:
    PHASE_EPISODES = [2, 2, 2] 
    MAX_STEPS = 5
else:
    PHASE_EPISODES = [80, 70, 50] 
    MAX_STEPS = 15

# ─── CORE FUNCTIONS ─────────────────────────────────────────

def setup_wandb():
    wandb.init(
        project="hf-hub-orchestrator",
        name="ppo-native-training",
        config={
            "model": "Qwen/Qwen2.5-1.5B-Instruct",
            "algorithm": "PPO (Native HF)",
            "total_episodes": sum(PHASE_EPISODES)
        }
    )

def load_model():
    print("Loading model natively via BitsAndBytes 4-bit...")
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    
    lora_config = LoraConfig(
        r=16,
        lora_alpha=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

    # 1. Load the Base Model correctly mapped to the GPU (Do NOT use "auto")
    base_model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-1.5B-Instruct",
        quantization_config=bnb_config,
        device_map="cuda" 
    )

    # 2. CRITICAL FIX: Prepare for 4-bit gradients
    base_model = prepare_model_for_kbit_training(base_model)

    # 3. Apply LoRA explicitly
    peft_model = get_peft_model(base_model, lora_config)
    
    # ─── NEW: THE GRADIENT LIFELINE ─────────────────────────────────
    # This forces PyTorch to track gradients through the frozen 4-bit layers
    peft_model.enable_input_require_grads()
    # ────────────────────────────────────────────────────────────────

    # 4. Wrap with the PPO Value Head
    model = AutoModelForCausalLMWithValueHead(peft_model)
    model.is_peft_model = True 

    # ─── RECONNECT THE BRAIN (POST-WRAPPER) ─────────────────────────
    # We must call this on the internal pretrained model so PyTorch sees it
    model.pretrained_model.enable_input_require_grads()
    
    # ─── NEW: UNFREEZE THE VALUE HEAD ───────────────────────────────
    # Ensure the newly attached Critic network is actually allowed to learn
    for name, param in model.named_parameters():
        if "v_head" in name:
            param.requires_grad = True
    # ────────────────────────────────────────────────────────────────
    
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    return model, tokenizer

def build_prompt(observation: dict, tokenizer) -> str:
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
    """
    Parses LLM output into a structured action.
    Indestructible: Will never crash the training loop, no matter what Qwen hallucinates.
    """
    import re
    try:
        # Attempt 1: Perfect JSON
        return json.loads(model_output.strip())
    except Exception:
        # Attempt 2: Qwen added conversational text, try to extract just the JSON block
        json_match = re.search(r'\{.*\}', model_output, re.DOTALL)
        if json_match:
            try:
                # We MUST wrap this in a try/except in case the extracted block uses 
                # single quotes or has syntax errors.
                return json.loads(json_match.group())
            except Exception:
                pass
                
    # Ultimate Fallback: Qwen completely failed to format.
    # We gracefully penalize it in the Gym instead of crashing the Python script.
    return {"action": "abort_pipeline", "instruction": "Format Error"}

# ─── PPO TRAINING ENGINE ────────────────────────────────────

def run_episode(model, tokenizer, env, chaos_enabled=False):
    from environment.models import Action
    
    env.chaos_enabled = chaos_enabled
    obs = env.reset()
    episode_data = []
    total_reward = 0.0
    done = False
    
    while not done and env._state.steps_used < MAX_STEPS:
        prompt = build_prompt(obs.model_dump(), tokenizer)
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs, 
                max_new_tokens=128, 
                temperature=0.7, 
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        resp_tensor = outputs[0][inputs['input_ids'].shape[1]:]
        response_text = tokenizer.decode(resp_tensor, skip_special_tokens=True)

        # ─── NEW: X-RAY VISION DEBUGGER ─────────────────────────────────
        print(f"\n[DEBUG] Raw LLM Output: {response_text.strip()[:100]}...")
        # ────────────────────────────────────────────────────────────────
        
        action_dict = parse_action(response_text)

        # ─── NEW: X-RAY VISION DEBUGGER ─────────────────────────────────
        print(f"[DEBUG] Parsed Action: {action_dict}")
        # ────────────────────────────────────────────────────────────────

        try:
            action = Action(action=action_dict.get("action", "abort_pipeline"), instruction=action_dict.get("instruction", ""))
        except:
            action = Action(action="abort_pipeline", instruction="Format Error")
            
        result = env.step(action)
        
        episode_data.append({
            "query": inputs['input_ids'][0],
            "response": resp_tensor,
            # CRITICAL FIX: Push the reward to the GPU, otherwise PyTorch crashes
            "reward": torch.tensor(result.reward, dtype=torch.float, device="cuda")
        })
        
        total_reward += result.reward
        obs = result.observation
        done = result.done
        
    return episode_data, total_reward


def train_phase(model, tokenizer, trainer, env, phase_name, episodes, chaos_enabled, start_ep):
    print(f"\n{'='*40}\nSTARTING {phase_name} ({episodes} episodes)\n{'='*40}")
    
    step_buffer = [] 

    for ep in range(episodes):
        episode_data, total_reward = run_episode(model, tokenizer, env, chaos_enabled)
        step_buffer.extend(episode_data)
        
        while len(step_buffer) >= trainer.config.batch_size:
            batch = step_buffer[:trainer.config.batch_size]
            step_buffer = step_buffer[trainer.config.batch_size:]
            
            queries = [step["query"] for step in batch]
            responses = [step["response"] for step in batch]
            rewards = [step["reward"] for step in batch]
            
            try:
                trainer.step(queries, responses, rewards)
            except Exception as e:
                print(f"⚠️ [WARNING] Math Error caught on batch. Skipping to save loop. Error: {e}")
                torch.cuda.empty_cache()
                continue
            
        global_ep = start_ep + ep
        wandb.log({"episode": global_ep, "episode_reward": total_reward, "phase": phase_name})
        print(f"Phase {phase_name} | Episode {global_ep} | Reward: {total_reward:.2f}")

    os.makedirs(f"checkpoints/{phase_name}", exist_ok=True)
    model.save_pretrained(f"checkpoints/{phase_name}")
    tokenizer.save_pretrained(f"checkpoints/{phase_name}")


def main():
    setup_wandb()
    model, tokenizer = load_model()
    
    config = PPOConfig(
        learning_rate=5e-6,
        batch_size=8,                  # The total pool of steps to collect
        mini_batch_size=1,             # Only 1 step in VRAM at a time (Max memory safety)
        gradient_accumulation_steps=8, # Accumulate 8 times to stabilize the learning
        max_grad_norm=0.5, 
        optimize_cuda_cache=True,
    )
    # Force garbage collection before training
    import gc
    gc.collect()
    torch.cuda.empty_cache()


    trainer = PPOTrainer(config=config, model=model, tokenizer=tokenizer)
    
    from environment.env import HFOrchestratorEnv
    env = HFOrchestratorEnv()
    
    try:
        train_phase(model, tokenizer, trainer, env, "Phase1_Happy", PHASE_EPISODES[0], False, 0)
        train_phase(model, tokenizer, trainer, env, "Phase2_Hard", PHASE_EPISODES[1], False, PHASE_EPISODES[0])
        train_phase(model, tokenizer, trainer, env, "Phase3_Chaos", PHASE_EPISODES[2], True, PHASE_EPISODES[0]+PHASE_EPISODES[1])
        
        print("\n✅ Training Complete! Saving final weights...")
        os.makedirs("checkpoints/final", exist_ok=True)
        model.save_pretrained("checkpoints/final")
        tokenizer.save_pretrained("checkpoints/final")
        
    except (KeyboardInterrupt, Exception) as e:
        print(f"\n🚨 [CRITICAL] Training Interrupted: {e}")
        print("💾 INITIATING EMERGENCY WEIGHT SAVE...")
        os.makedirs("checkpoints/emergency_save", exist_ok=True)
        model.save_pretrained("checkpoints/emergency_save")
        tokenizer.save_pretrained("checkpoints/emergency_save")
        print("✅ Emergency save complete. Progress secured.")
        
    finally:
        wandb.finish()

if __name__ == "__main__":
    main()