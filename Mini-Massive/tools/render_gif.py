import json, sys, numpy as np, matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from pathlib import Path

run_dir = Path(sys.argv[1])   # e.g., out/run_0001
out_gif = Path(sys.argv[2])   # e.g., flock.gif

frames = [json.loads(l) for l in open(run_dir/"frames.jsonl")]
W = max(a["x"] for fr in frames for a in fr["agents"])
H = max(a["y"] for fr in frames for a in fr["agents"])

fig, ax = plt.subplots()
ax.set_xlim(0, W); ax.set_ylim(0, H); ax.set_aspect("equal"); ax.axis("off")
first = np.array([[a["x"], a["y"]] for a in frames[0]["agents"]])
scat = ax.scatter(first[:,0], first[:,1], s=12)

def upd(i):
    xy = np.array([[a["x"], a["y"]] for a in frames[i]["agents"]])
    scat.set_offsets(xy); return scat,

anim = FuncAnimation(fig, upd, frames=len(frames), interval=33, blit=True)
anim.save(out_gif, writer=PillowWriter(fps=30))
print(f"[ok] saved {out_gif}")