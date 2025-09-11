import numpy as np
from minimassive.world import SpatialHash
from minimassive.agents import Agent

def test_neighbors_basic():
    sh = SpatialHash(cell=10.0)
    agents = [Agent(i, np.array([i*2.0, 0.0]), np.array([0.0,0.0])) for i in range(5)]
    sh.rebuild(agents)
    nbs = sh.neighbors(np.array([4.0, 0.0]), radius=5.0)
    # Should include self vicinity: ids 1 and 2 likely within radius
    ids = sorted(a.id for a in nbs)
    assert 1 in ids and 2 in ids