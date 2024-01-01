
class Model(torch.nn.Module):
    def __init__(self, max_value: float, min_value: float):
        super().__init__()
        self.linear = torch.nn.Linear(64 * 64, 64 * 64)
        self.maxValue = max_value
        self.minValue = min_value
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = torch.clamp_min(v1, min_value=self.minValue)
        v3 = torch.clamp_min(v2, min_value=self.minValue)
        return v3

# Initializing the model
m = Model(3.5, 1.5)

# Inputs to the model
x1 = torch.randn(1, 64 * 64)
