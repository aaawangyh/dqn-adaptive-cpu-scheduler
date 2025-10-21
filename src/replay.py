
from collections import deque, namedtuple
import random, numpy as np
Transition = namedtuple("Transition", "state action reward next_state done priority")

class PrioritizedReplay:
    def __init__(self, capacity=100000, alpha=0.6, seed=0):
        self.capacity = capacity
        self.alpha = alpha
        self.rng = random.Random(seed)
        self.buf = deque(maxlen=capacity)

    def push(self, s,a,r,ns,d,priority=1.0):
        p = max(1e-3, float(priority)) ** self.alpha
        self.buf.append(Transition(s,a,r,ns,d,p))

    def __len__(self): return len(self.buf)

    def sample(self, batch):
        if len(self.buf) < batch: return None
        pri = np.array([tr.priority for tr in self.buf], dtype=np.float64)
        probs = pri/pri.sum()
        idxs = np.random.choice(len(self.buf), size=batch, replace=False, p=probs)
        batch = [self.buf[i] for i in idxs]
        isw = (len(self.buf)*probs[idxs])**(-1); isw = isw/isw.max()
        return batch, isw, idxs

    def update_priorities(self, idxs, new_p):
        for i,p in zip(idxs, new_p):
            tr = self.buf[i]
            self.buf[i] = Transition(tr.state,tr.action,tr.reward,tr.next_state,tr.done,max(1e-3,float(p)))
