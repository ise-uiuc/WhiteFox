
class Model(torch.nn.Module):
    def __init__(self, min_value=-0.9, max_value=0.8):
        super().__init__()
        self.linear = torch.nn.Linear(6, 1)
        self.min_value = min_value
        self.max_value = max_value

    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = torch.clamp_min(v1, self.min_value)
        v3 = torch.clamp_max(v2, self.max_value)
        return v3

# Initializing the model
min_value = random.random()
max_value = min_value + random.random()
m = Model(min_value=min_value, max_value=max_value)

# Inputs to the model
x1 = torch.randn(1, 6)
