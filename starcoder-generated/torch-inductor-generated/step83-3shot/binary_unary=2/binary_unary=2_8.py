
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 256, 1, stride=1, padding=0)
        self.bn = torch.nn.BatchNorm2d(num_features=256, eps=0.001)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = self.bn(v1)
        v3 = v2 - 1000
        v4 = F.relu(v3)
        return v4
# Inputs to the model
x1 = torch.randn(1, 1, 32, 32)
