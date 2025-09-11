from __future__ import annotations
from .behaviors import separation, cohesion, alignment, lane_follow, limit

class Scheduler:
    def __init__(self, world, dt=0.05, mode="flock"):
        self.world = world
        self.dt = dt
        self.mode = mode

    def step(self):
        if self.mode == "flock":
            for a in self.world.agents:
                nbs = [nb for nb in self.world.grid.neighbors(a.pos, 20.0) if nb is not a]
                force = separation(a, nbs, desired_dist=8.0) \
                      + cohesion(a, nbs) \
                      + alignment(a, nbs)
                a.vel = limit(a.vel + force, a.max_speed)
        elif self.mode == "traffic":
            for a in self.world.agents:
                force = lane_follow(a, self.world.waypoints)
                nbs = [nb for nb in self.world.grid.neighbors(a.pos, 10.0) if nb is not a]
                force = force + separation(a, nbs, desired_dist=6.0, max_force=0.4)
                a.vel = limit(a.vel + force, a.max_speed)
        self.world.step(self.dt)