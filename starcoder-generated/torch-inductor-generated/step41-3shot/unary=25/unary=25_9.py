
class Model(torch.nn.Module):
    def __init__(self, negative_slope=0.01):
        self.negative_slope=negative_slope
        super().__init__()

    def forward(self, x1):
        v1 = torch.matmul(x1, linear_weight)
        v2 = v1 > 0
        v3 = v1 * self.negative_slope
        v4 = torch.where(v2, v1, v3)
        return v4

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(10, 10)
