
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(20, 63, 1, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(63, 20, 1, stride=1, padding=1)
        self.bn = torch.nn.BatchNorm2d(20)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = torch.relu(v1)
        v3 = self.conv2(v2)
        v4 = torch.relu(v3)
        v5 = self.bn(v4)
        return v5
# Inputs to the model
x1 = torch.randn(1, 20, 4, 4)
