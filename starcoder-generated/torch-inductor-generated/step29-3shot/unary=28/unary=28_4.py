
class Model(torch.nn.Module):
    def __init__(self, min=-1.0, max=1.0):
        super().__init__()
        # This is a linear transformation
        self.linear = torch.nn.Linear(64, 128)
        # min and max are provided as keyword arguments
        self.min = min
        self.max = max
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = torch.clamp_min(v1, self.min)
        v3 = torch.clamp_max(v2, self.max)
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 64)
