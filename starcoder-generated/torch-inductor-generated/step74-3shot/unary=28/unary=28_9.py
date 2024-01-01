
class Model(torch.nn.Module):
    def __init__(self, minmax):
        super().__init__()
        self.linear = torch.nn.Linear(5, 3)
        min_value = minmax.get("min", -1.0)
        max_value = minmax.get("max", 1.0)
        self.min_value = min_value
        self.max_value = max_value
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = v1.clamp_min(self.min_value)
        v3 = v2.clamp_max(self.max_value)
        return v3

# Initializing the model
minmax = {"min": 0.0, "max": 5.0}
m = Model(minmax)

# Inputs to the model
x1 = torch.randn(1, 5)
