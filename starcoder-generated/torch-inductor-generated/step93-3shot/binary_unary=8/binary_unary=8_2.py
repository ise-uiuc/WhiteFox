
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.bn1 = torch.nn.BatchNorm2d(num_features=1)
        self.bn2 = torch.nn.BatchNorm2d(num_features=16)
        self.conv1 = torch.nn.Conv2d(1, 16, 3, stride=1, padding=0)
    def forward(self, x1):
        v1 = self.bn1(x1)
        v2 = torch.relu(self.conv1(x1))
        v3 = self.bn2(v2)
        v4 = v1 + v3
        v5 = torch.relu(self.conv1(x1))
        v6 = v4 + v5
        return v6
# Inputs to the model
x1 = torch.randn(1, 1, 64, 64)
