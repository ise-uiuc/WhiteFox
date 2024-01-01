
class Model(torch.nn.Module):
    def __init__(self, negative_slope):
        super().__init__()
        self.linear = torch.nn.Linear(32, 16, bias=False)
        self.negative_slope = negative_slope
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = v1 > 0
        v3 = v1 * self.negative_slope
        v4 = torch.where(v2, v1, v3)
        return v4

# Initializing the model with negative slope 0.5
m1 = Model(0.5)

# Initializing the model with negative slope 0.1
m2 = Model(0.1)

# Inputs to the models
x1 = torch.randn(5, 32)
__output1__ = m1(x1)
__output2__ = m2(x1)

