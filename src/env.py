
from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Dict

@dataclass
class Task:
    id: int
    remaining: float
    deadline: float
    arrival_t: int
    priority: int = 0

@dataclass
class Core:
    task_id: int = -1
    time_on_task: int = 0

class CpuSchedEnv:
    def __init__(self, cores=4, max_queue=32, dvfs_levels=(0.7,1.0), preemption_cost=0.0,
                 slice_ticks=1, max_steps=2000, thermal_tau=0.99, arrival_rate=0.6,
                 task_exec_mean=20, task_exec_std=8, deadline_slack_mean=15, deadline_slack_std=6,
                 seed=0):
        self.rng = np.random.default_rng(seed)
        self.c = cores
        self.cores = [Core() for _ in range(cores)]
        self.max_queue = max_queue
        self.dvfs_levels = list(dvfs_levels)
        self.dvfs_idx = len(self.dvfs_levels)-1
        self.preemption_cost = preemption_cost
        self.slice = slice_ticks
        self.max_steps = max_steps
        self.thermal_tau = thermal_tau
        self.arrival_rate = arrival_rate
        self.exec_mean = task_exec_mean
        self.exec_std = task_exec_std
        self.deadline_slack_mean = deadline_slack_mean
        self.deadline_slack_std = deadline_slack_std
        self.reset()

    def reset(self):
        self.step_t = 0
        self.queue: List[Task] = []
        self.next_task_id = 1
        self.temperature = 0.0
        self.last_ctx_switches = 0
        self.done = False
        self.info: Dict = {"deadline_miss":0, "latency_sum":0.0, "finished":0, "energy":0.0}
        for core in self.cores:
            core.task_id = -1
            core.time_on_task = 0
        for _ in range(self.c*2):
            self._maybe_arrive(force=True)
        return self._state()

    def step(self, action: Tuple[List[int], int]):
        """Action: (dispatch_ids_for_each_core, dvfs_index)"""
        if self.done:
            raise RuntimeError("Call reset() before step()")

        self.last_ctx_switches = 0
        # DVFS
        self.dvfs_idx = int(np.clip(action[1], 0, len(self.dvfs_levels)-1))
        speed = self.dvfs_levels[self.dvfs_idx]

        # Dispatch
        dispatch = action[0]
        for i, core in enumerate(self.cores):
            target_id = dispatch[i]
            if target_id == -2:  # keep
                continue
            if core.task_id != target_id:
                if core.task_id != -1 and target_id != -1:
                    self.last_ctx_switches += 1
                core.task_id = target_id
                core.time_on_task = 0

        # Execute one tick
        finished = []
        for core in self.cores:
            if core.task_id == -1: 
                continue
            t = self._find(core.task_id)
            if t is None:
                core.task_id = -1
                continue
            t.remaining -= speed * self.slice
            core.time_on_task += 1
            if t.remaining <= 0:
                finished.append(t.id)
                core.task_id = -1

        deadline_miss = 0
        latency_sum = 0.0
        for tid in finished:
            t = self._pop(tid)
            if t is None: 
                continue
            latency_sum += (self.step_t - t.arrival_t)
            if self.step_t > t.deadline:
                deadline_miss += 1
        self.info["finished"] += len(finished)
        self.info["latency_sum"] += latency_sum
        self.info["deadline_miss"] += deadline_miss

        # Thermal + energy
        util = sum(1 for c in self.cores if c.task_id != -1) / self.c
        self.temperature = self.thermal_tau*self.temperature + (1-self.thermal_tau)*(util*speed)
        energy = util*(0.8+0.2*speed) + 0.01*self.temperature
        self.info["energy"] += energy

        # Arrivals and overflow
        self._maybe_arrive()
        if len(self.queue) > self.max_queue:
            self.queue = self.queue[:self.max_queue]

        avg_latency = (latency_sum/max(1,len(finished))) if finished else 0.0
        miss = 1.0 if deadline_miss>0 else 0.0
        reward = -avg_latency - 4.0*miss - 0.5*energy - 0.2*self.last_ctx_switches - self.preemption_cost*self.last_ctx_switches

        self.step_t += 1
        if self.step_t >= self.max_steps:
            self.done = True
        return self._state(), float(reward), self.done, {"queue_len":len(self.queue), **self.info}

    # ---- helpers ----
    def _maybe_arrive(self, force=False):
        k = 1 if force else np.random.poisson(self.arrival_rate)
        for _ in range(k):
            exec_t = max(1.0, float(self.rng.normal(self.exec_mean, self.exec_std)))
            slack = max(1.0, float(self.rng.normal(self.deadline_slack_mean, self.deadline_slack_std)))
            deadline = self.step_t + exec_t + slack
            self.queue.insert(0, Task(self.next_task_id, exec_t, deadline, self.step_t))
            self.next_task_id += 1

    def _find(self, tid:int):
        for t in self.queue:
            if t.id == tid:
                return t
        return None

    def _pop(self, tid:int):
        for i,t in enumerate(self.queue):
            if t.id == tid:
                return self.queue.pop(i)
        return None

    def _state(self):
        topk = 8
        q = sorted(self.queue, key=lambda t: t.deadline)[:topk]
        q_feats = []
        for t in q:
            q_feats.extend([t.remaining/100.0, (t.deadline - self.step_t)/100.0])
        q_feats += [0.0]*(topk*2 - len(q)*2)

        core_feats = []
        for c in self.cores:
            core_feats.extend([1.0 if c.task_id!=-1 else 0.0, c.time_on_task/50.0])

        dvfs_onehot = [0.0]*len(self.dvfs_levels)
        dvfs_onehot[self.dvfs_idx] = 1.0
        state = np.array(q_feats + core_feats + dvfs_onehot + [self.temperature], dtype=np.float32)
        return state

    def enumerate_actions(self):
        actions = []
        M = min(8, len(self.queue))
        candidates = [t.id for t in sorted(self.queue, key=lambda t: t.deadline)[:M]]
        # patterns
        keep_all = [-2]*self.c
        idle_all = [-1]*self.c
        edf = (candidates + [-1]*self.c)[:self.c]
        actions.extend([ (keep_all, d) for d in range(len(self.dvfs_levels)) ])
        actions.extend([ (idle_all, d) for d in range(len(self.dvfs_levels)) ])
        actions.extend([ (edf, d) for d in range(len(self.dvfs_levels)) ])
        # fill idle cores only
        pat = []
        for core in self.cores:
            if core.task_id == -1 and candidates:
                pat.append(candidates.pop(0))
            else:
                pat.append(-2)
        actions.extend([ (pat, d) for d in range(len(self.dvfs_levels)) ])
        return actions
