
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.bn1 = torch.nn.BatchNorm2d(16)
        self.conv1 = torch.nn.Conv2d(3, 16, 1, stride=1, padding=1)
        self.bn2 = torch.nn.BatchNorm2d(16)
    def forward(self, x1):
        v1 = torch.nn.functional.relu(self.bn1(self.conv1(x1)))
        v2 = torch.nn.functional.relu(self.bn2(self.conv1(x1)))
        v3 = v1
        v4 = torch.nn.functional.relu(self.bn2(self.conv1(v3)))
        v5 = v1 + v2 + v4
        v6 = torch.nn.functional.relu(self.bn1(self.conv1(v5)))
        return v6
# Inputs to the model
x1 = torch.randn(1, 3, 224, 224)
