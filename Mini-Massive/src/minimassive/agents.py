from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np

@dataclass
class Agent:
    id: int
    pos: np.ndarray           # shape (2,)
    vel: np.ndarray           # shape (2,)
    max_speed: float = 2.0
    radius: float = 1.5
    kind: str = "generic"     # "bird", "car", etc.
    state: str = "idle"
    log_data: dict = field(default_factory=dict)

    def step(self, world, dt: float):
        speed = np.linalg.norm(self.vel)
        if speed > 1e-6:
            self.vel = self.vel / speed * min(speed, self.max_speed)
        self.pos = self.pos + self.vel * dt

        w, h = world.width, world.height
        if self.pos[0] < 0 or self.pos[0] > w:
            self.vel[0] *= -1
            self.pos[0] = np.clip(self.pos[0], 0, w)
        if self.pos[1] < 0 or self.pos[1] > h:
            self.vel[1] *= -1
            self.pos[1] = np.clip(self.pos[1], 0, h)