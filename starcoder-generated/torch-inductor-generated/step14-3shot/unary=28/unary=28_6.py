
import torch.nn.functional as F

class Model(torch.nn.Module):
    def __init__(self, min_value=-9.181805824296979e+37, max_value=9.152457547119236e+37):
        super().__init__()
        self.fc = torch.nn.Linear(3, 4)
        self.min_value = min_value
        self.max_value = max_value
 
    def forward(self, x1):
        v1 = self.fc(x1)
        v2 = torch.clamp_min(v1, self.min_value)
        return torch.clamp_max(v2, self.max_value)

# Initializing the model
m = Model(1, 5)

# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
