
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 5, stride=2, padding=1, dilation=3)
        self.bn = torch.nn.BatchNorm2d(8)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = self.bn(v1)
        v3 = v2 - 0.5
        v4 = F.relu(v3)
        v5 = torch.squeeze(v4, 0)
        return v5
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
