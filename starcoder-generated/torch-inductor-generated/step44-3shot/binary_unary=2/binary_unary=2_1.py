
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(256, 256, 1, stride=1, padding=1)
        self.bn1 = torch.nn.BatchNorm2d(256)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = v1 - 128.5
        v3 = F.relu(v2)
        v4 = self.bn1(v3)
        v5 = v4 - 128.5
        v6 = F.relu(v5)
        return v6
# Inputs to the model
x1 = torch.randn(1, 256, 8, 8)
