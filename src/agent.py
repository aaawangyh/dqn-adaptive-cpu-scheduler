
import math, random
import numpy as np
import torch, torch.nn as nn, torch.optim as optim
from .models import MLP
from .replay import PrioritizedReplay

class DQNAgent:
    def __init__(self, state_dim, n_actions, cfg):
        self.gamma = cfg.get("gamma", 0.99)
        self.lr = cfg.get("lr", 2.5e-4)
        self.batch = cfg.get("batch_size", 256)
        self.tgt_up = cfg.get("target_update", 2000)
        self.double = cfg.get("double_dqn", True)
        self.dueling = cfg.get("dueling", True)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.n_actions = n_actions
        self.online = MLP(state_dim, n_actions, dueling=self.dueling).to(self.device)
        self.target = MLP(state_dim, n_actions, dueling=self.dueling).to(self.device)
        self.target.load_state_dict(self.online.state_dict())

        self.opt = optim.Adam(self.online.parameters(), lr=self.lr)
        self.replay = PrioritizedReplay(cfg.get("buffer_capacity", 200000), cfg.get("per_alpha", 0.6))

        eps = cfg.get("eps", {"start":1.0, "end":0.05, "decay":50000})
        self.e0, self.e1, self.decay = eps["start"], eps["end"], eps["decay"]
        self.step_t = 0

    def act(self, state):
        eps = self.e1 + (self.e0 - self.e1) * math.exp(-self.step_t / self.decay)
        self.step_t += 1
        if random.random() < eps:
            return random.randrange(self.n_actions)
        with torch.no_grad():
            s = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            q = self.online(s)
            return int(q.argmax(1).item())

    def push(self, s,a,r,ns,d,priority=1.0):
        self.replay.push(s,a,r,ns,d,priority)

    def step(self):
        sample = self.replay.sample(self.batch)
        if sample is None: return None
        batch, isw, idxs = sample
        states = torch.tensor([b.state for b in batch], dtype=torch.float32, device=self.device)
        actions = torch.tensor([b.action for b in batch], dtype=torch.long, device=self.device)
        rewards = torch.tensor([b.reward for b in batch], dtype=torch.float32, device=self.device)
        next_states = torch.tensor([b.next_state for b in batch], dtype=torch.float32, device=self.device)
        dones = torch.tensor([b.done for b in batch], dtype=torch.float32, device=self.device)
        isw = torch.tensor(isw, dtype=torch.float32, device=self.device)

        q = self.online(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            if self.double:
                na = self.online(next_states).argmax(1)
                qn = self.target(next_states).gather(1, na.unsqueeze(1)).squeeze(1)
            else:
                qn = self.target(next_states).max(1)[0]
            target = rewards + (1 - dones) * self.gamma * qn

        td = target - q
        loss = (isw * td.pow(2)).mean()
        self.opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), 10.0)
        self.opt.step()

        new_p = (td.detach().abs().cpu().numpy() + 1e-3)
        self.replay.update_priorities(idxs, new_p)

        if self.step_t % self.tgt_up == 0:
            self.target.load_state_dict(self.online.state_dict())
        return float(loss.item())
