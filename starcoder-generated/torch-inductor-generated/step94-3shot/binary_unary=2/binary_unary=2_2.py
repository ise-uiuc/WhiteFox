
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 16, 3, stride=2, padding=1)
        self.bn = torch.nn.BatchNorm2d(16)
    def forward(self, x3):
        v1 = self.conv(x3)
        v2 = torch.max(v1, dim=1, keepdim=False)[0]
        v3 = v2 - 2
        v4 = F.relu(v3)
        v5 = self.bn(v4)
        v6 = v5 - 3
        v7 = F.relu(v6)
        return v7
# Inputs to the model
x3 = torch.randn(1, 3, 64, 64)
