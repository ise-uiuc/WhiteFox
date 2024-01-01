
class Model(torch.nn.Module):
    def __init__(self, min_value=None, max_value=None):
        super().__init__()
        self.linear = torch.nn.Linear(3, 8, bias=0)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = torch.clamp_min(v1, min_value)
        v3 = torch.clamp_max(v2, max_value)
        return v3

# Initializing the model
m = Model(min_value, max_value)

# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
