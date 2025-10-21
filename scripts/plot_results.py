
import argparse, json, os
import matplotlib.pyplot as plt

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    args = ap.parse_args()
    for p in args.runs:
        with open(p,"r") as f:
            data = json.load(f)
        if "reward" in data:
            plt.figure()
            plt.plot(data["reward"])
            plt.title("Reward over Episodes")
            plt.xlabel("Episode"); plt.ylabel("Reward")
            out = os.path.basename(os.path.dirname(p))+"_reward.png"
            plt.tight_layout(); plt.savefig(out); print("Saved", out)
        else:
            plt.figure()
            names = list(data.keys()); vals = [data[k]["reward"] for k in names]
            plt.bar(names, vals)
            plt.title("Aggregate Reward")
            plt.ylabel("Reward")
            out = "baselines_reward.png"
            plt.tight_layout(); plt.savefig(out); print("Saved", out)

if __name__ == "__main__":
    main()
