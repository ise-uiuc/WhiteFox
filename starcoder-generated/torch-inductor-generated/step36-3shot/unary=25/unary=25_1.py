
class Model(torch.nn.Module):
    def __init__(self, negative_slope=0.01):
        super().__init__()
        self.linear = torch.nn.Linear(3, 8)

    def forward(self, x2):
        v1 = self.linear(x2)
        v2 = v1 > 0
        v3 = v1 * negative_slope
        v4 = torch.where(v2, v1, v3)
        return v4

# Initializing the model
m1 = Model(0.01)
m2 = Model(0.001)

# Inputs to the model
x2 = torch.randn(1, 3)
