
# Adaptive Task Scheduling in Multi‑Core CPUs Using Deep Q‑Networks (DQN)

Research‑grade, config‑driven repo for a learning‑based CPU scheduler:
- Gym‑like simulator of multi‑core CPUs with DVFS and thermal proxy
- DQN agent (Double & Dueling), prioritized replay
- Baselines: RR, EDF, SRTF
- Reproducible experiments + plotting

## Quick Start
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m experiments.run_experiment --config configs/default.yaml --tag dqn_default
python -m src.evaluate --checkpoint artifacts/dqn_default/best.pt --episodes 50
python scripts/plot_results.py --runs artifacts/dqn_default/metrics.json artifacts/baselines/metrics.json
```
