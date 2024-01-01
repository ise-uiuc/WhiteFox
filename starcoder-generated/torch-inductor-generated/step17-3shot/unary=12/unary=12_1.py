
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 3, 3, stride=1, padding=1, groups=3)
        self.conv1 = torch.nn.Conv2d(3, 8, 11, stride=2, padding=5)
        self.conv2 = torch.nn.Conv2d(128, 128, 3, stride=1, padding=1, groups=4)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = self.conv1(v1)
        v3 = torch.sigmoid(v2)
        v4 = v3 * v2
        v5 = self.conv2(v4)
        v6 = torch.sigmoid(v5)
        v7 = v6 * v4
        return v7
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
