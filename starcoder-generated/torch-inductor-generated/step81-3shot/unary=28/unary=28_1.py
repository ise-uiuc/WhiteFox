
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(64, 32)
 
    def forward(self, x1, __arg1__):
        v1 = self.linear(x1)
        v2 = torch.clamp_min(v1, __arg1__)
        v3 = torch.clamp_max(v2, __arg2__)
        return v3

# Initializing the model
m = Model()

# Inputs to the model and constants
x1 = torch.randn(1, 64)
__arg1__ = 0.8
__arg2__ = 0.8
