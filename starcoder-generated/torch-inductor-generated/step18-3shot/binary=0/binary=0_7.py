
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(13, 13, 1, stride=1, padding=1)
        self.bn = torch.nn.BatchNorm2d(13)
        self.conv_ = torch.nn.Conv2d(13, 13, 1, stride=1)
        self.relu = torch.nn.ReLU()
    def forward(self, x1, padding1=None):
        v1 = self.conv(x1)
        v2 = self.bn(v1)
        v3 = torch.cat((v1, v2), axis=1)
        v4 = self.conv_(v3)
        v5 = self.relu(v2)
        v6 = v5 + v4
        return v6
# Inputs to the model
x1 = torch.randn(1, 13, 64, 64)
