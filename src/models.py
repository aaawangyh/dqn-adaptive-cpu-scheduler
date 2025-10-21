
import torch, torch.nn as nn

class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden=(256,256), dueling=True):
        super().__init__()
        self.dueling = dueling
        self.trunk = nn.Sequential(
            nn.Linear(in_dim, hidden[0]), nn.ReLU(),
            nn.Linear(hidden[0], hidden[1]), nn.ReLU(),
        )
        if dueling:
            self.adv = nn.Sequential(nn.Linear(hidden[1], hidden[1]), nn.ReLU(), nn.Linear(hidden[1], out_dim))
            self.val = nn.Sequential(nn.Linear(hidden[1], hidden[1]), nn.ReLU(), nn.Linear(hidden[1], 1))
        else:
            self.head = nn.Linear(hidden[1], out_dim)

    def forward(self, x):
        z = self.trunk(x)
        if self.dueling:
            a = self.adv(z)
            v = self.val(z)
            return v + a - a.mean(dim=1, keepdim=True)
        return self.head(z)
