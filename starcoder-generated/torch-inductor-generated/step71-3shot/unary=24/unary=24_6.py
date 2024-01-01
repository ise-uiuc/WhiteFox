
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = torch.nn.MaxPool2d(3, stride=1, padding=7)
    def forward(self, x):
        negative_slope = 3.933803
        v1 = self.pool(x)
        v2 = v1 > 0
        v3 = v1 * negative_slope
        v4 = torch.where(v2, v1, v3)
        return v4
# Inputs to the model
x1 = torch.randn(1, 3, 5, 7)
