
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 3, 2, stride=2, padding=0)
        self.bn1 = torch.nn.BatchNorm2d(3)
        self.conv2 = torch.nn.Conv2d(3, 3, 2, stride=2, padding=0)
        self.bn2 = torch.nn.BatchNorm2d(3)
        self.conv3 = torch.nn.Conv2d(3, 2, 2, stride=2, padding=0)
        self.bn3 = torch.nn.BatchNorm2d(2)
    def forward(self, x1):
        v1 = self.bn1(self.conv1(x1))
        v2 = torch.nn.functional.relu(v1)
        v3 = self.bn2(self.conv2(v2))
        v4 = torch.nn.functional.relu(v3)
        v5 = self.bn3(self.conv3(v4))
        v6 = torch.nn.functional.relu(v5)
        return v6
# Inputs to the model
x1 = torch.randn(1, 3, 256, 256)
