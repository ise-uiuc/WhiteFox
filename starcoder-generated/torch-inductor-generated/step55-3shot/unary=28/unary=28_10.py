
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x1):
        v1 = F.linear(x1, weight, bias)
        v2 = torch.clamp_min(v1, min_value)
        v3 = torch.clamp_max(v2, max_value)
        return v3

# Initializing the model
m = Model()

# Inputs to model
x1 = torch.randn(1, 3)
weight = torch.randn(2, 3)
bias = torch.randn(2)
min_value = 0.5
max_value = 2.0
