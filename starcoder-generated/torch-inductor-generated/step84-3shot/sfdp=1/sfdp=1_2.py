

from torch.nn import Dropout
from torch import mm

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout = Dropout(p=0.2)
        self.dense = torch.nn.Linear(12, 16)
        self.dense_norm = torch.nn.LayerNorm(16)
 
    def forward(self, x1, x2):
        v1 = mm(x1, x2.transpose(0, 1))
        x3 = self.dropout(v1)
        v2 = self.dense(x3)
        v2 = self.dense_norm(v2)
        return v2

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(2, 12)
x2 = torch.randn(12, 2)
