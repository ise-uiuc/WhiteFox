
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(64 * 64 * 3, 100)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = torch.clamp_min(v1, min_value=128.0)
        v3 = torch.clamp_max(v2, max_value=64.0)
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
_output = m(x1)

