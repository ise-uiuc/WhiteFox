
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(64, 64, 3, stride=2, padding=1)
        self.conv2 = torch.nn.Conv2d(64, 32, 3, stride=2, padding=1)
        self.conv3 = torch.nn.Conv2d(32, 3, 3)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = v1 - 64
        v3 = F.relu(v2)
        v4 = self.conv2(v3)
        v5 = v4 - 32
        v6 = F.relu(v5)
        v7 = self.conv3(v6)
        v8 = v7 - 456
        return v8
# Inputs to the model
x1 = torch.randn(1, 64, 56, 56)
