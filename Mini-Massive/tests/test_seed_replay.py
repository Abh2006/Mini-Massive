import numpy as np
from minimassive.cli import build_world_from_scenario
from minimassive.utils import set_seed

def test_seed_replay_deterministic():
    scn = {"mode":"flock","width":100,"height":80,"agents":10,"dt":0.1}
    set_seed(7)
    w1, s1 = build_world_from_scenario(scn)
    set_seed(7)
    w2, s2 = build_world_from_scenario(scn)
    p1 = np.array([a.pos for a in w1.agents])
    p2 = np.array([a.pos for a in w2.agents])
    assert np.allclose(p1, p2)