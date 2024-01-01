
def clamp_fn(min_value, max_value):
    def clamp(x):
        return torch.clamp_max(torch.clamp_min(x, min_value), max_value)
    return clamp

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 8)
        self.clamp = clamp_fn(min_value=-0.5, max_value=0.5)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = self.clamp(v1)
        return v2

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3)
