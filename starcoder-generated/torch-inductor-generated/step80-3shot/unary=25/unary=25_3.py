
class LeakyReLUModel(torch.nn.Module):
    def __init__(self, negative_slope):
        super().__init__()
        self.linear = torch.nn.Linear(128, 128, bias=True)
        self.negative_slope = negative_slope
       
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = v1 > 0
        v3 = v1 * self.negative_slope
        v4 = torch.where(v2, v1, v3)
        return v4

# Initializing the model for negative slope of 1
m1 = LeakyReLUModel(1)

# Initializing the model for negative slope of 0.1
m01 = LeakyReLUModel(0.1)

# Inputs to the model
x1 = torch.randn(1, 128)
__output1__ = m1(x1)
__output01__ = m01(x1)

