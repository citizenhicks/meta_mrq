# MR.Q: Towards General-Purpose Model-Free Reinforcement Learning

This repository contains the implementation of MR.Q, an extension of MR.Q for model-free deep reinforcement learning (RL). The method is inspired by the paper:

- 📄 [Towards General-Purpose Model-Free Reinforcement Learning](https://arxiv.org/pdf/2501.16142)
- 📍 Published at ICLR 2025
- ✍️ Scott Fujimoto, Pierluca D’Oro, Amy Zhang, Yuandong Tian, Michael Rabbat (Meta FAIR)

- 🔗 Original MR.Q implementation by Meta FAIR: [facebookresearch/MRQ](https://github.com/facebookresearch/MRQ)

## 📌 Overview
MR.Q aims to unify model-free deep RL across diverse benchmarks by leveraging model-based representations while avoiding the computational overhead of traditional model-based RL. It achieves competitive performance across 118 environments with a single set of hyperparameters.

Key highlights of MR.Q:

- Model-based representations for approximate value function linearization.
- No need for planning or simulated trajectories.
- Generalization across diverse RL benchmarks (continuous/discrete actions, vector/pixel observations).
- Efficient training compared to model-based baselines.


## How to run
This is a simplified version of the Meta original implementation specifically for gymansium environments. The repo uses uv for package management, hence you can use the following steps to run the code:

```bash
uv venv
source venv/bin/activate
uv sync
uv python main.py 
```