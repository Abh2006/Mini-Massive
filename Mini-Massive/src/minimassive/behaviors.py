from __future__ import annotations
import numpy as np

def limit(v: np.ndarray, max_len: float) -> np.ndarray:
    n = np.linalg.norm(v)
    if n <= max_len or n < 1e-9:
        return v
    return v * (max_len / n)

def seek(agent, target: np.ndarray, gain=1.0, max_force=0.5):
    desired = target - agent.pos
    return limit(desired * gain, max_force)

def separation(agent, neighbors, desired_dist=8.0, max_force=0.6):
    steer = np.zeros(2)
    count = 0
    for nb in neighbors:
        d = np.linalg.norm(agent.pos - nb.pos)
        if 1e-6 < d < desired_dist:
            steer += (agent.pos - nb.pos) / d
            count += 1
    if count:
        steer /= count
    return limit(steer, max_force)

def alignment(agent, neighbors, max_force=0.3):
    if not neighbors:
        return np.zeros(2)
    avg = sum(nb.vel for nb in neighbors) / len(neighbors)
    return limit(avg - agent.vel, max_force)

def cohesion(agent, neighbors, max_force=0.3):
    if not neighbors:
        return np.zeros(2)
    center = sum(nb.pos for nb in neighbors) / len(neighbors)
    return limit(center - agent.pos, max_force)

def lane_follow(agent, waypoints, gain=0.8, max_force=0.5):
    if not waypoints:
        return np.zeros(2)
    dists = [np.linalg.norm(agent.pos - wp) for wp in waypoints]
    idx = int(np.argmin(dists))
    target = waypoints[min(idx + 1, len(waypoints) - 1)]
    return seek(agent, target, gain=gain, max_force=max_force)