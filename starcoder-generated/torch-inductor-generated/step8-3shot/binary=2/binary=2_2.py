
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1, groups=3)
        self.conv1 = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1, groups=3)
    def forward(self, x1, x2):
        v1 = self.conv(x1)
        v2 = x2.view(1, 16, 100, 100)
        v3 = self.conv1(v2)
        v4 = v1 - v3
        return v4
# Inputs to the model
x1 = torch.randn(1, 3, 33, 33)
x2 = torch.randn(1, 3, 33, 33)
