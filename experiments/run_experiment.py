
import argparse, yaml, tempfile, os
from src.train import main as train_main

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()

    if args.tag:
        with open(args.config,"r") as f: cfg = yaml.safe_load(f)
        cfg["logging"]["tag"] = args.tag
        fd, tmp = tempfile.mkstemp(suffix=".yaml")
        with os.fdopen(fd, "w") as g:
            yaml.safe_dump(cfg, g)
        try:
            train_main(tmp)
        finally:
            os.unlink(tmp)
    else:
        train_main(args.config)
