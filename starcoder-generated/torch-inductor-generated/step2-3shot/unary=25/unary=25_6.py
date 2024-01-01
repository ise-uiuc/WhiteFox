
class Model(torch.nn.Module):
    def __init__(self, negative_slope: float = 0.01):
        super().__init__()
        self.linear = torch.nn.Linear(1024, 1024)
        self.negative_slope = negative_slope
 
    def forward(self, x1):
        v1 = self.linear(x1)
        return torch.where(v1 > 0, v1, v1 * self.negative_slope)

# Initializing the model
m = Model(0.01)

# Inputs to the model
x1 = torch.randn(1, 1024)
