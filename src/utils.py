
import os, json, random, numpy as np
def set_seed(seed:int):
    random.seed(seed); np.random.seed(seed)
def ensure_dir(p): os.makedirs(p, exist_ok=True)
def save_json(path, obj):
    with open(path, "w") as f: json.dump(obj, f, indent=2)
