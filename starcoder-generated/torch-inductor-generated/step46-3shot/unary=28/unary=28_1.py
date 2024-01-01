
class Model(torch.nn.Module):
    def __init__(self, min_value, max_value):
        super().__init__()
        self.linear = torch.nn.Linear(16, 1)
 
    def forward(self, x1):
        v0 = self.linear(x1)
        v1 = torch.clamp_min(v0, min_value)
        v2 = torch.clamp_max(v1, max_value)
        return v2

# Initializing the model
m = Model(min_value, max_value)

# Inputs to the model
x1 = torch.randn(1, 16)
