
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 8, 1, stride=1)
        self.conv2 = torch.nn.Conv2d(3, 8, 1, stride=1)
        self.bn1 = torch.nn.BatchNorm2d(8)
    def forward(self, x1):
        v1 = F.relu(self.conv1(x1))
        v2 = F.relu(self.conv2(v1))
        v3 = v1 + v2
        v4 = self.bn1(v3)
        return v4
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
