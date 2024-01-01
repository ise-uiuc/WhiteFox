
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x, y):
        negative_slope = 0.987091
        v1 = x > y
        v2 = x - y
        v3 = v1 * negative_slope
        v4 = torch.where(v2, x, v3)
        return v4 * x * y
# Inputs to the model
x1 = torch.randn(1, 3)
x2 = torch.randn(1, 3)
