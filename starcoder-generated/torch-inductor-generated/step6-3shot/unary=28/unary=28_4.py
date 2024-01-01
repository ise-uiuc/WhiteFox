
class Model(torch.nn.Module):
    def __init__(self, min_value, max_value):
        super().__init__()
 
    def forward(self, x1):
        v1 = torch.nn.functional.linear(x1, self.weight, self.bias)
        v2 = torch.clamp_min(v1, self.min_value)
        v3 = torch.clamp_max(v2, self.max_value)
        return v3

# Initializing the model
m = Model(min_value=-1.0, max_value=1.0)

# Inputs to the model
x1 = torch.randn(1, 16)
