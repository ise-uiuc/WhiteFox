
class Model(torch.nn.Module):
    def __init__(self, min_value, max_value):
        super().__init__()
        self.min_value = min_value
        self.max_value = max_value

    def forward(self, x1):
        v1 = torch.nn.Linear(32, 64)(x1)
        v2 = torch.clamp_min(v1, self.min_value)
        v3 = torch.clamp_max(v2, self.max_value)
        return v3

# Initializing the model
m = Model(min_value=-0.2853724, max_value=1.1921649)

# Inputs to the model
x1 = torch.randn(1, 32)
