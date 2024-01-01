
class Model(torch.nn.Module):
    def __init__(self, min_value=-3.0):
        super().__init__()
 
    def forward(self, x1):
        v1 = torch.nn.functional.linear(x1, m.weight, None)
        v2 = v1.clamp_min(self.min_value)
        v3 = v2.clamp_max(-1.5)
        return v3

# Initializing the model
m = Model(1.5)

# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
