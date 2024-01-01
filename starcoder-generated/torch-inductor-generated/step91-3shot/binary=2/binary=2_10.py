
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
    def forward(self, x2, x1):
        v1 = self.conv(x1)
        v2 = self.conv(x2)
        v3 = v1 - v2
        return v3
# Inputs to the model
x2 = torch.randn(1, 3, 64, 64)
x1 = torch.randn(1, 3, 64, 64)
