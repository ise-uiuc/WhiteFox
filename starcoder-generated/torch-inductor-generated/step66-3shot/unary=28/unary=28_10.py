
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(196, 43)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = torch.clamp_min(v1, min=-10.0)
        v3 = torch.clamp_max(v2, max=10.0)
        return v3

# Initializing the model
m = Model()

# Inputs to the model
__minimum_value__ = -10.0
__maximum_value__ = 10.0
x1 = torch.randn(6, 3, 7, 7)
