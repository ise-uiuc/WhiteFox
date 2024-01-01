
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        negative_slope = 0.1
        v1 = torch.nn.functional.relu6(x) * negative_slope
        v2 = v1 > 0
        v3 = v1 * negative_slope
        v4 = torch.where(v2, x, v3)
        return v4
# Inputs to the model
x1 = torch.randn(1, 5, 31, 31)
