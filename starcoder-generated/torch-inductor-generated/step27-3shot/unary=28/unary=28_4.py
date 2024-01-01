
class Model(torch.nn.Module):
    def __init__(self, a, b, negative_slope):
        super().__init__()
        self.linear = torch.nn.Linear(3, 50, bias=True)
        self.min = min_value
        self.max = max_value
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = torch.clamp_min(v1, min_value=self.min)
        v3 = torch.clamp_max(v2, max_value=self.max)
        return v3

# Initializing the model
m = Model()

# Inputs to the model
min_value = -0.1727
max_value = 0.5057
x1 = torch.randn(1, 3, 64, 64)
