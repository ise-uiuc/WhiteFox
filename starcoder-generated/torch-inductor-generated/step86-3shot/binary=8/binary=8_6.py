
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 8, 1, stride=1, padding=0)
        self.bn = torch.nn.BatchNorm2d(8)
    def forward(self, x1):
        v1 = self.bn(self.conv1(x1))
        v2 = self.bn(self.conv1(x1))
        v3 = torch.relu(v1)
        v4 = self.bn(torch.relu(v2))
        v5 = F.relu(v1)
        v6 = F.relu(v2)
        v7 = torch.add(v1, v3)
        v8 = torch.add(v5, v4)
        v9 = torch.add(v7, v8)
        return torch.sigmoid(v9)
# Inputs to the model
x1 = torch.randn(2, 3, 300, 300)
