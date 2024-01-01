
import torch.nn as nn
class Model(nn.Module):
    def __init__(self):
        super().__init__()
 
    def split_forward(self, x):
        split = x.split(1, 1)
        return split
 
    def forward(self, x1):
        torch.cat(self.split_forward(x1), dim=1)
     
 
# Initializing the model
m = Model()
 
# Inputs to the model
x1 = torch.randn(1, 10, 1, 2)
