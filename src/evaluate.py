
import os, json, yaml, torch, numpy as np
from .env import CpuSchedEnv
from .models import MLP
from .heuristics import rr_policy, edf_policy, srtf_policy

def eval_policy(env, fn, episodes):
    out = {"reward":[], "miss":[], "avg_lat":[]}
    for _ in range(episodes):
        s = env.reset(); done=False; ep_r=0.0; info={}
        while not done:
            dispatch, dvfs = fn(env)
            s, r, done, info = env.step((dispatch, dvfs))
            ep_r += r
        fin = info.get("finished", 1)
        out["reward"].append(ep_r)
        out["miss"].append(info.get("deadline_miss",0)/max(1,fin))
        out["avg_lat"].append(info.get("latency_sum",0)/max(1,fin))
    return {k: float(np.mean(v)) for k,v in out.items()}

def eval_dqn(env, ckpt, episodes, dueling=True):
    s = env.reset(); n_actions = len(env.enumerate_actions())
    net = MLP(len(s), n_actions, dueling=dueling); net.load_state_dict(torch.load(ckpt, map_location="cpu")); net.eval()
    out = {"reward":[], "miss":[], "avg_lat":[]}
    for _ in range(episodes):
        s = env.reset(); done=False; ep_r=0.0; info={}
        while not done:
            acts = env.enumerate_actions()
            with torch.no_grad():
                q = net(torch.tensor(s).unsqueeze(0))
                a = int(q.argmax(1).item()); a = min(a, len(acts)-1)
            s, r, done, info = env.step(acts[a]); ep_r += r
        fin = info.get("finished", 1)
        out["reward"].append(ep_r)
        out["miss"].append(info.get("deadline_miss",0)/max(1,fin))
        out["avg_lat"].append(info.get("latency_sum",0)/max(1,fin))
    return {k: float(np.mean(v)) for k,v in out.items()}

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/small.yaml")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--episodes", type=int, default=20)
    args = ap.parse_args()

    with open(args.config,"r") as f:
        cfg = yaml.safe_load(f)
    e = cfg["env"]
    env = CpuSchedEnv(
        cores=e["cores"], max_queue=e["max_queue"], dvfs_levels=tuple(e["dvfs_levels"]),
        preemption_cost=e["preemption_cost"], slice_ticks=e["slice"], max_steps=e["max_steps"],
        thermal_tau=e["thermal_tau"], arrival_rate=e["arrival_rate"],
        task_exec_mean=e["task"]["exec_mean"], task_exec_std=e["task"]["exec_std"],
        deadline_slack_mean=e["task"]["deadline_slack_mean"], deadline_slack_std=e["task"]["deadline_slack_std"],
        seed=cfg.get("seed",0)
    )

    os.makedirs("artifacts/baselines", exist_ok=True)
    if args.checkpoint:
        dqn = eval_dqn(env, args.checkpoint, args.episodes)
    else:
        dqn = {"reward": float("nan"), "miss": float("nan"), "avg_lat": float("nan")}
    rr = eval_policy(env, rr_policy, args.episodes)
    edf = eval_policy(env, edf_policy, args.episodes)
    srtf = eval_policy(env, srtf_policy, args.episodes)

    res = {"DQN": dqn, "RR": rr, "EDF": edf, "SRTF": srtf}
    with open("artifacts/baselines/metrics.json","w") as f: json.dump(res, f, indent=2)
    print(json.dumps(res, indent=2))
