
class Model(torch.nn.Module):
    def __init__(self, minValue=0.01, maxValue=0.99):
        super().__init__()
        self.linear = torch.nn.Linear(1, 1)

    def forward(self, x1):
        v1 = self.linear(x1)
        v1 = torch.clamp_min(v1, min_value=minValue)
        return torch.clamp_max(v1, max_value=maxValue)

# Initializing the model
m = Model(minValue=0.02, maxValue=0.85)

# Inputs to the model
x1 = torch.randn(1, 1)
