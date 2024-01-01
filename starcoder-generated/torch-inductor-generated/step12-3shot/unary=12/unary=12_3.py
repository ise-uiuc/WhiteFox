
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = torch.nn.ReLU()
        self.conv1x1 = torch.nn.Conv2d(3, 6, 1, stride=1, padding=1)
        self.conv3x3 = torch.nn.Conv2d(6, 12, 3, stride=1, padding=1)
        self.conv1x1_2 = torch.nn.Conv2d(12, 16, 1, stride=1, padding=1)
    def forward(self, x1):
        v1 = self.relu(x1)
        v2 = self.conv1x1(v1)
        v3 = self.relu(v2)
        v4 = self.conv3x3(v3)
        v5 = self.relu(v4)
        v6 = self.conv1x1_2(v5)
        v7 = F.sigmoid(v6)
        v8 = v6.mul(v7)
        return v8
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
