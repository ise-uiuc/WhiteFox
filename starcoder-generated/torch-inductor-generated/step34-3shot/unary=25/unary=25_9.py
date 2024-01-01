
class Model(torch.nn.Module):
    def __init__(self, negative_slope):
        super().init()
        self.negative_slope = negative_slope
        self.linear = torch.nn.Linear(3, 8)

    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = v1 > 0
        v3 = v1 * self.negative_slope
        v4 = torch.where(v2, v1, v3)
        return v4

# Initializing a model
m = Model(negative_slope=0.42)

# Input to the model
x1 = torch.randn(84, 3, 64, 64)
