
class Model(torch.nn.Module):
    def __init__(self, negative_slope):
        super().__init__()
        self.negative_slope = negative_slope
        self.linear = torch.nn.Linear(32, 64)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        t = v1 > 0
        v2 = v1 * self.negative_slope
        v3 = torch.where(t, v1, v2)
        return v3

# Initializing the model with negative slope
m = Model(-0.1)

# Inputs to the model
x1 = torch.randn(1, 32)
