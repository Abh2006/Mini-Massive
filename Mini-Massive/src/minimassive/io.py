from __future__ import annotations
import json
from pathlib import Path

def save_run(frames, out_dir: str):
    p = Path(out_dir)
    p.mkdir(parents=True, exist_ok=True)
    meta = {"frames": len(frames), "count": len(frames[0]["agents"]) if frames else 0}
    with open(p / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    with open(p / "frames.jsonl", "w") as f:
        for fr in frames:
            serial = {
                "t": fr["t"],
                "agents": [{"id": a["id"], "x": float(a["x"]), "y": float(a["y"])} for a in fr["agents"]]
            }
            f.write(json.dumps(serial) + "\n")

def load_scenario(path: str):
    with open(path, "r") as f:
        return json.load(f)