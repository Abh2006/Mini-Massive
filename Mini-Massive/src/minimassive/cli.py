from __future__ import annotations
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from pathlib import Path

from .utils import set_seed
from .agents import Agent
from .world import World
from .scheduler import Scheduler
from .io import save_run, load_scenario

def build_world_from_scenario(scn) -> tuple[World, Scheduler]:
    width = scn.get("width", 200)
    height = scn.get("height", 120)
    n = scn.get("agents", 300)
    mode = scn.get("mode", "flock")

    agents = []
    for i in range(n):
        pos = np.array([np.random.uniform(0, width), np.random.uniform(0, height)], dtype=float)
        vel = np.random.uniform(-1, 1, size=2)
        kind = scn.get("agent_kind", "bird" if mode == "flock" else "car")
        max_speed = 2.5 if kind == "bird" else 2.0
        agents.append(Agent(i, pos, vel, max_speed=max_speed, kind=kind))

    waypoints = [np.array([float(x), float(y)], dtype=float) for x, y in scn.get("waypoints", [])]
    world = World(width=width, height=height, agents=agents, waypoints=waypoints)
    sched = Scheduler(world, dt=scn.get("dt", 0.05), mode=mode)
    return world, sched

def run_headless(scn_path: str, out_dir: str, duration: float, seed: int | None):
    set_seed(seed)
    scn = load_scenario(scn_path)
    world, sched = build_world_from_scenario(scn)

    frames = []
    t = 0.0
    dt = scn.get("dt", 0.05)
    steps = int(duration / dt)
    for _ in range(steps):
        sched.step()
        frames.append({
            "t": t,
            "agents": [{"id": a.id, "x": a.pos[0], "y": a.pos[1]} for a in world.agents]
        })
        t += dt
    save_run(frames, out_dir)
    print(f"[ok] wrote {len(frames)} frames to {out_dir}")

def run_live(scn_path: str, duration: float, seed: int | None, save_path: str | None):
    set_seed(seed)
    scn = load_scenario(scn_path)
    world, sched = build_world_from_scenario(scn)

    fig, ax = plt.subplots()
    ax.set_xlim(0, world.width); ax.set_ylim(0, world.height); ax.set_aspect("equal")
    speeds = [ (a.vel[0]**2 + a.vel[1]**2)**0.5 for a in world.agents ]
    scat = ax.scatter([a.pos[0] for a in world.agents],
                      [a.pos[1] for a in world.agents],
                      s=20, c=speeds, cmap="viridis")
    cb = fig.colorbar(scat, ax=ax); cb.set_label("speed")

    dt = scn.get("dt", 0.05)
    frames = int(duration / dt)

    def update(_):
        sched.step()
        offs = [[a.pos[0], a.pos[1]] for a in world.agents]
        scat.set_offsets(offs)
        spd = [ (a.vel[0]**2 + a.vel[1]**2)**0.5 for a in world.agents ]
        import numpy as _np
        scat.set_array(_np.array(spd))
        return scat,

    anim = FuncAnimation(fig, update, frames=frames, interval=dt*1000, blit=True)
    if save_path:
        anim.save(save_path, writer=PillowWriter(fps=int(1/dt)))
        print(f"[ok] saved {save_path}")
    else:
        plt.show()

def render_run(indir: str, outfile: str, fps: int = 30):
    import json
    p = Path(indir)
    frames_path = p / "frames.jsonl"
    meta_path = p / "meta.json"
    if not frames_path.exists() or not meta_path.exists():
        raise SystemExit(f"[err] missing frames/meta in {indir}")

    frames = []
    with open(frames_path) as f:
        for line in f:
            fr = json.loads(line)
            frames.append(fr)

    W = max(a["x"] for fr in frames for a in fr["agents"])
    H = max(a["y"] for fr in frames for a in fr["agents"])

    fig, ax = plt.subplots()
    ax.set_xlim(0, W); ax.set_ylim(0, H); ax.set_aspect("equal"); ax.axis("off")

    import numpy as np
    first = np.array([[a["x"], a["y"]] for a in frames[0]["agents"]])
    scat = ax.scatter(first[:, 0], first[:, 1], s=12)

    def upd(i):
        xy = np.array([[a["x"], a["y"]] for a in frames[i]["agents"]])
        scat.set_offsets(xy)
        return scat,

    anim = FuncAnimation(fig, upd, frames=len(frames), interval=1000/fps, blit=True)
    anim.save(outfile, writer=PillowWriter(fps=fps))
    print(f"[ok] saved {outfile}")

def main():
    ap = argparse.ArgumentParser(prog="mini-massive", description="Mini-Massive CLI")
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="Run simulation with live visualization")
    r.add_argument("--scenario", required=True, help="path to scenario JSON")
    r.add_argument("--duration", type=float, default=30.0)
    r.add_argument("--seed", type=int)
    r.add_argument("--save", help="save animation to GIF (e.g., flock.gif)")

    h = sub.add_parser("headless", help="Run simulation and write frames to out dir")
    h.add_argument("--scenario", required=True)
    h.add_argument("--duration", type=float, default=30.0)
    h.add_argument("--out", default="out/run_0001")
    h.add_argument("--seed", type=int)

    v = sub.add_parser("render", help="Convert a headless run directory to a GIF")
    v.add_argument("--in",  dest="indir",   required=True, help="dir with meta.json & frames.jsonl")
    v.add_argument("--out", dest="outfile", default="run.gif", help="output GIF path")
    v.add_argument("--fps", type=int, default=30, help="GIF frames per second")

    args = ap.parse_args()
    if args.cmd == "run":
        run_live(args.scenario, args.duration, args.seed, args.save)
    elif args.cmd == "headless":
        run_headless(args.scenario, args.out, args.duration, args.seed)
    elif args.cmd == "render":
        render_run(args.indir, args.outfile, fps=args.fps)

if __name__ == "__main__":
    main()