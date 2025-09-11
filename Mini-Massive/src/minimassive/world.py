from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
from .agents import Agent

@dataclass
class SpatialHash:
    cell: float
    table: dict = field(default_factory=dict)

    def _key(self, p):
        return (int(p[0] // self.cell), int(p[1] // self.cell))

    def rebuild(self, agents):
        self.table.clear()
        for a in agents:
            k = self._key(a.pos)
            self.table.setdefault(k, []).append(a)

    def neighbors(self, pos, radius):
        cx, cy = self._key(pos)
        out = []
        cells = range(-1, 2)
        r2 = radius * radius
        for dx in cells:
            for dy in cells:
                for a in self.table.get((cx + dx, cy + dy), []):
                    if np.sum((a.pos - pos) ** 2) <= r2:
                        out.append(a)
        return out

@dataclass
class World:
    width: float
    height: float
    agents: list[Agent]
    waypoints: list[np.ndarray] = field(default_factory=list)
    grid: SpatialHash = field(default_factory=lambda: SpatialHash(16.0))

    def step(self, dt: float):
        self.grid.rebuild(self.agents)
        for a in self.agents:
            a.step(self, dt)