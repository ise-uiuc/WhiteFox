
import torch
import torch.nn 

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 1)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = torch.sigmoid(v1)
        return v2
 
# Initializing the model
m = Model()
 
# Getting a list of the model's parameters
print("Model parameters: %s\n" % m.parameters())
 
# Inputs to the model
x1 = torch.randn(1, 3)
