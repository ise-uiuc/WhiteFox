
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(5, 10, 1, stride=1, padding=1)
    def forward(self, x1, x2, x3):
        v1 = self.conv(x1)
        v2 = self.conv(x2)
        v3 = self.conv(x3)
        v4 = v1 + v2
        v5 = v4 + v3
        return v5
# Inputs to the model
x1 = torch.randn(1, 5, 64, 64)
x2 = torch.randn(1, 5, 64, 64)
x3 = torch.randn(1, 5, 64, 64)
