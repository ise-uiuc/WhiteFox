
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 3, 1, stride=1, padding=1, groups=3)
        self.conv2 = torch.nn.Conv2d(3, 6, 3, stride=1, padding=1, groups=6)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = self.conv2(v1)
        return v2
# Inputs to the model
x1 = torch.randn(1, 1, 64, 64)
