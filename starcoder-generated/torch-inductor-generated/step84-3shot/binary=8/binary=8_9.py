
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(8, 8, 1, stride=1, padding=1)
        self.bn1 = torch.nn.BatchNorm2d(8)
        self.relu = torch.nn.ReLU()
        self.bn2 = torch.nn.BatchNorm2d(8)

    def forward(self, x1):
        v1 = self.bn1(self.conv1(x1))
        v2 = self.relu(self.conv2(v1))
        v3 = self.bn2(v2)
        v4 = self.conv1(v3)
        v5 = v1 + v4
        return v5
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
