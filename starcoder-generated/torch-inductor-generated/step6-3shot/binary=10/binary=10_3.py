
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(64, 64)
 
    def forward(self, x1, x2):
        v1 = self.linear(x1)
        v2 = v1 + x2
        return v2

# Initializing the model
m = Model()

import numpy as np

from IPython.display import display

# Inputs to the model
x1 = torch.randn(1, 64)
x2 = torch.Tensor(np.ones([1, 64]))
