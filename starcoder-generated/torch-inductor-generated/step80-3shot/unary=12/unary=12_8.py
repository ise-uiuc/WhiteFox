
class Model(n.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 8, 1, groups=1, padding=1)
        self.conv2 = torch.nn.Conv2d(8, 32, 3, groups=8, padding=1)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = F.sigmoid(v1)
        v3 = v1 * v2
        v4 = self.conv2(v3)
        return v2 + v4
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
