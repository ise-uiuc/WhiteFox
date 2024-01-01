
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(32, 64)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        __minimum_value__ = 0
        __maximum_value__ = 6
        v2 = torch.clamp(v1 + 3, min=__minimum_value__, max=__maximum_value__)
        v3 = v2 / 6
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 32)
