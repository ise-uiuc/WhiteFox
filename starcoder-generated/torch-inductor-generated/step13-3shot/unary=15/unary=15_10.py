
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 32, (3, 3), stride=1, padding=1)
        self.bn1 = torch.nn.BatchNorm2d(32)
        self.conv2 = torch.nn.Conv2d(32, 32, (3, 3), stride=2, padding=1)
        self.bn2 = torch.nn.BatchNorm2d(32)
    def forward(self, x1):
        v6 = self.conv1(x1)
        v7 = self.bn1(v6)
        v8 = torch.relu(v7)
        v9 = self.conv2(v8)
        v10 = self.bn2(v9)
        v11 = torch.relu(v10)
        return v11
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
