
class Model(torch.nn.Module):
    def __init__(self, min_value, max_value):
        super().__init__()
        self.linear = Linear(5, 3)
        self.min_value = min_value
        self.max_value = max_value
 
    def forward(self, x1):
        v0 = self.linear(x1)
        v1 = torch.clamp_min(v0, self.min_value)
        v2 = torch.clamp_max(v1, self.max_value)
        return v2

# Initializing the model
m = Model(0, 1)

# Inputs to the model
x1 = torch.randn(1, 5, 64, 64)
