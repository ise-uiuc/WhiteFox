
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.bn1 = torch.nn.BatchNorm2d(1)
        self.bn2 = torch.nn.BatchNorm2d(16)
        self.bn3 = torch.nn.BatchNorm2d(32)
        self.conv1 = torch.nn.Conv2d(1, 8, 1, stride=1, padding=0)
        self.conv2 = torch.nn.Conv2d(16, 16, 1, stride=1, padding=0)
        self.conv3 = torch.nn.Conv2d(32, 32, 1, stride=1, padding=0)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = torch.relu(self.bn1(v1))
        v3 = v2.reshape(4, 8, 8)
        v4 = v3.unsqueeze(2)
        v5 = self.conv2(v4)
        v6 = v5.reshape(4, 16, 4)
        v7 = v6.unsqueeze(2)
        v8 = self.conv3(v7)
        v9 = v8.reshape(4, 32, 2)
        v10 = v9.unsqueeze(2)
        return v10
# Inputs to the model
x1 = torch.randn(64, 1, 4, 2)
