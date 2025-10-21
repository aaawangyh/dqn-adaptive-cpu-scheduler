
import os, yaml, torch
from .env import CpuSchedEnv
from .agent import DQNAgent
from .utils import set_seed, ensure_dir, save_json

def main(cfg_path):
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)
    seed = cfg.get("seed", 0)
    set_seed(seed)

    e = cfg["env"]
    env = CpuSchedEnv(
        cores=e["cores"], max_queue=e["max_queue"], dvfs_levels=tuple(e["dvfs_levels"]),
        preemption_cost=e["preemption_cost"], slice_ticks=e["slice"], max_steps=e["max_steps"],
        thermal_tau=e["thermal_tau"], arrival_rate=e["arrival_rate"],
        task_exec_mean=e["task"]["exec_mean"], task_exec_std=e["task"]["exec_std"],
        deadline_slack_mean=e["task"]["deadline_slack_mean"], deadline_slack_std=e["task"]["deadline_slack_std"],
        seed=seed,
    )
    state = env.reset()
    actions = env.enumerate_actions()
    agent = DQNAgent(len(state), len(actions), cfg["agent"])

    out_dir = os.path.join(cfg["logging"]["out_dir"], cfg["logging"]["tag"])
    ensure_dir(out_dir)
    metrics = {"loss":[], "reward":[], "miss_ratio":[], "avg_latency":[]}

    total = cfg["agent"]["train_steps"]; warm = cfg["agent"]["start_learning"]
    step = 0; best = -1e9
    while step < total:
        s = env.reset(); done=False; ep_reward=0.0; ep_fin=0; ep_lat=0.0; info={}
        while not done and step < total:
            actions = env.enumerate_actions()
            a = min(agent.act(s), len(actions)-1)
            ns, r, done, info = env.step(actions[a])
            agent.push(s, a, r, ns, float(done), priority=abs(r)+1e-3)
            s = ns; ep_reward += r; step += 1
            ep_fin = info["finished"]; ep_lat = info["latency_sum"]
            if step > warm:
                loss = agent.step()
                if loss is not None: metrics["loss"].append(loss)
        avg_lat = (ep_lat/max(1, ep_fin))
        miss = info["deadline_miss"]/max(1, info["finished"])
        metrics["reward"].append(ep_reward); metrics["avg_latency"].append(avg_lat); metrics["miss_ratio"].append(miss)
        if ep_reward > best:
            best = ep_reward
            torch.save(agent.online.state_dict(), os.path.join(out_dir, "best.pt"))
        if len(metrics["reward"]) % 10 == 0:
            save_json(os.path.join(out_dir, "metrics.json"), metrics)

    save_json(os.path.join(out_dir, "metrics.json"), metrics)

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    main(ap.parse_args().config)
