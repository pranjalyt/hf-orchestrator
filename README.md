---
title: HF Orchestrator Env
emoji: 🚀
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

# 🚀 HF Hub Orchestrator (Meta OpenEnv)

An enterprise-grade Reinforcement Learning environment built for the **Meta PyTorch OpenEnv Hackathon**. This environment trains Large Language Models to autonomously navigate the Hugging Face Hub, evaluate models, and architect production-ready ML pipelines under strict hardware and budget constraints.

**Team:** TensorTitans

## 🧠 Architecture Overview

This project implements a multi-agent orchestration Gym designed for GRPO/PPO training via TRL and Unsloth.

* **State-Locking & Anti-Hacking:** Built-in safeguards against endless loops, reckless QA skipping, and hallucinated LLM actions.
* **Cost-Aware RL (Enterprise ROI):** Custom reward shaping that mathematically forces the agent to balance pipeline accuracy against simulated cloud compute costs (e.g., favoring `INT4` quantization for edge deployments).
* **Curriculum Learning:** 3-phase training escalation (Happy Path → Hardware Limits → API Chaos/503 Errors).

## 🛠️ The Stack
* **Environment:** OpenEnv + FastAPI + Docker
* **RL Training:** Hugging Face TRL (PPOTrainer)
* **Model:** Qwen 2.5 1.5B (Instruct) via Unsloth (4-bit LoRA)
* **Tracking:** Weights & Biases

## 💻 Running the Environment Locally

If you want to test the environment rules without running the massive training loop:

```bash
# Install environment dependencies
pip install -r requirements.txt

# Run the test suite
python scripts/test_episode.py