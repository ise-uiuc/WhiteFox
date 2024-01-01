
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 10)
 
    def forward(self, v1, v2):
        t1 = self.linear(v1)
        t2 = t1 - v2
        return t2

# Initializing the model
import torch.nn

m = Model()

# Inputs to the model
v1 = torch.randn(1, 10)
v2 = torch.randn(1, 10)
