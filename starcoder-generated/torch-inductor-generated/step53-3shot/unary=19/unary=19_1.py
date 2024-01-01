
from torch.nn.functional import softplus
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 8)
 
    def forward(self, x):
        v = self.linear(x)
        v = v * 0.5
        v = torch.sigmoid(v)
        return torch.nn.functional.softplus(v)
 
# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(1, 3)
