
import numpy as np
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        t1 = F.dropout(x1, p=0.5, training=int(np.array([1, 0, 4, 4, 1])!=1))
        return t1
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        t1 = F.dropout(x1, p=0.5, training=self.training)
        return t1
# Inputs to the model
x1 = torch.randn(1, 20)
