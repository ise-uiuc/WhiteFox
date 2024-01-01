
class Model(torch.nn.Module):
    def __init__(self, negative_slope):
        super().__init__()
        self.linear = torch.nn.Linear(16, 32)
        self.negative_slope = float(negative_slope)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = v1 > 0
        v3 = v1 * self.negative_slope
        v4 = torch.where(v2, v1, v3)
        return v4

# Initializing the model
n = Model(0.2)

# Inputs to the model
x1 = torch.randn(1, 16, 32, 32)
