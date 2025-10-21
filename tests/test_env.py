
from src.env import CpuSchedEnv
def test_env():
    env = CpuSchedEnv(seed=0, max_steps=10)
    s = env.reset()
    assert s is not None and len(s)>0
    acts = env.enumerate_actions()
    ns, r, done, info = env.step(acts[0])
    assert isinstance(r, float)
