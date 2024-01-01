
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 32, 3, stride=stride1, padding=1)
        self.bn1 = torch.nn.BatchNorm2d(32)
        self.conv1_1 = torch.nn.Conv2d(32, 32, 3, stride=stride2, padding=1)
        self.bn1_2 = torch.nn.BatchNorm2d(32)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = self.bn1(v1)
        v3 = torch.relu(v2)
        v4 = self.conv1_1(v3)
        v5 = self.bn1_2(v4)
        v6 = torch.relu(v5)
        return v6
# Inputs to the model
x1 = torch.randn(1, 3, 256, 256)
