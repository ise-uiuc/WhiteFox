
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(10, 5)
 
    def forward(self, x1, x2):
        v1 = self.fc(x1)
        v2 = v1 + x2
        return v2

# Initializing the model
m = Model()

# Inputs to the model
import numpy as np
x1 = torch.randn(1, 10)
x2 = torch.from_numpy(np.ones(5, dtype=np.float32))
