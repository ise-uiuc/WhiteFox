
import torch
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(128, 128)
 
    def forward(self, x2):
        v1 = self.linear(x2)
        v2 = torch.tanh(v1)
        return v1
 
# Initializing the model
m = Model()

# Inputs to the model
x2 = torch.randn(1, 128, device='cuda')
