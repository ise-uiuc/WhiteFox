
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(1)
        self.conv = nn.Conv2d(2, 2, 1)
        torch.manual_seed(1)
        self.bn = nn.BatchNorm2d(2)
        torch.manual_seed(1)
        self.bn1 = nn.BatchNorm2d(2, affine=False)
    def forward(self, x3):
        v3 = self.conv(x3)
        v4 = self.conv(v3)
        v4a = self.bn(v4 + 1.)
        v4a = self.conv(v4a)
        v4a = self.bn(torch.relu((v4) + 5))
        v4a = self.conv(v4a)
        v4a = self.bn(v4a + v4)
        return v4a
# Inputs to the model
x3 = torch.randn(1, 2, 3, 3)
