
class Model(torch.nn.Module):
    def __init__(self, min_value, max_value):
        super().__init__()
        self.linear = torch.nn.Linear(3, 8)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        return torch.clamp_max(torch.clamp_min(v1, min_value), max_value)

# Initializing the model
m = Model(min_value=-1.0, max_value=2.0)

# Inputs to the model
x1 = torch.randn(1, 3)
