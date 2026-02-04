import torch
from torch import nn

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            # first layer (64 units)
            nn.LazyLinear(64), nn.ReLU(),
            # first hidden layer (32 units)
            nn.LazyLinear(32), nn.LazyBatchNorm1d(), nn.ReLU(), nn.Dropout(),
            # second hidden layer (16 units)
            nn.LazyLinear(16), nn.LazyBatchNorm1d(), nn.ReLU(), nn.Dropout(),
            # third hidden layer (8 units)
            nn.LazyLinear(8), nn.LazyBatchNorm1d(), nn.ReLU(), nn.Dropout(),
            # final output layer
            nn.LazyLinear(1))
    
    def forward(self, x):
        return self.net(x)