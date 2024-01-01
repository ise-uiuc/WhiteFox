
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x, negative_slope=1):
        v1 = x > 0
        v2 = x * negative_slope
        v3 = torch.where(v1, x, v2)
        return v3
# Inputs to the model
x1 = torch.randn(1, 16, 64, 64)
x = 1
