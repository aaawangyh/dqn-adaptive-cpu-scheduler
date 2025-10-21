
from .env import CpuSchedEnv

def rr_policy(env: CpuSchedEnv):
    dispatch = []
    q = list(reversed(env.queue))
    for _ in env.cores:
        dispatch.append(q.pop().id if q else -1)
    return dispatch, env.dvfs_idx

def edf_policy(env: CpuSchedEnv):
    q = sorted(env.queue, key=lambda t: t.deadline)
    dispatch = [(q[i].id if i < len(q) else -1) for i,_ in enumerate(env.cores)]
    return dispatch, env.dvfs_idx

def srtf_policy(env: CpuSchedEnv):
    q = sorted(env.queue, key=lambda t: t.remaining)
    dispatch = [(q[i].id if i < len(q) else -1) for i,_ in enumerate(env.cores)]
    return dispatch, env.dvfs_idx
