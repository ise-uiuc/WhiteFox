
class Model(torch.nn.Module):
    def __init__(self, negative_slope):
        super().__init__()
        self.linear = torch.nn.Linear(10, 2)
        self.negative_slope = negative_slope

    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = v1 > 0
        v3 = torch.empty_like(v1)
        v3.fill_(self.negative_slope)
        v3 = v1 * v3
        v4 = torch.where(v2, v1, v3)
        return v4

# Initializing the model with negative slopes 0.01 and 0.02 respectively.
m1 = Model(0.01)
m2 = Model(0.02)

# Inputs to the model
x1 = torch.randn(1, 10)
__out1 = m1(x1)
__out2 = m2(x1)

