
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1, bias=False)
        self.bn = torch.nn.BatchNorm2d(8)
    def forward(self, x1):
        v1 = self.bn(self.conv(x1))
        v2 = torch.sigmoid(v1)
        return v1 * v2
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
