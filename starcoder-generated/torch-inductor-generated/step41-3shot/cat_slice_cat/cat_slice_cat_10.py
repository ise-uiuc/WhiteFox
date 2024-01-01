
import sys
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x1):
   return x1

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
x2 = torch.randn(1, 3, 64, 64)
