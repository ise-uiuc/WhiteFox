
class Model(torch.nn.Module):
    def __init__(self, min_value, max_value):
        super().__init__()
        self.linear = torch.nn.Linear(32, 8)

    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = torch.clamp_min(v1, min=min_value)
        v3 = torch.clamp_max(v2, max=max_value)
        return v3

# Initializing the model
m1 = Model(min_value=0.01, max_value=-0.01)
m2 = Model(min_value=-0.005, max_value=0.005)
m3 = Model(min_value=-0.01, max_value=0.01)

# Inputs to the model
x1 = torch.randn(1, 32, 64, 64)
x2 = torch.randn(1, 32, 64, 64)
# For m1 and m2, the outputs are close to 0 or very close to 0. 
# For m3, the outputs should be in the range of (-1, 1).
