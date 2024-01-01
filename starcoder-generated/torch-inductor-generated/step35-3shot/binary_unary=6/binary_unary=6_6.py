
import torch
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(224*224*3, 1000)
 
    def forward(self, x1):
        t1 = self.linear(x1)
        t2 = t1 - 0.2360
        t3 = torch.nn.ReLU()(t2)
        return t3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3 * 224 * 224)
