
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(32, 32, 1, stride=1, padding=0, dilation=1)
        self.conv2 = torch.nn.Conv2d(32, 32, 1, stride=1, padding=0, dilation=2)
        self.conv3 = torch.nn.Conv2d(32, 32, 1, stride=1, padding=0, dilation=3)
    def forward(self, x1, x2, x3):
        v1 = self.conv1(x1)
        v2 = v1 + x2
        v3 = torch.relu(v2)
        v4 = self.conv2(v3)
        v5 = v4 + x2
        v6 = torch.relu(v5)
        a1 = self.conv3(v6)
        v7 = v3 + a1
        v8 = torch.relu(v7)
        v9 = self.conv1(v8)
        v10 = v9 + x3
        v11 = torch.relu(v10)
        return v11
# Inputs to the model
x1 = torch.randn(1, 32, 64, 64)
x2 = torch.randn(1, 32, 64, 64)
x3 = torch.randn(1, 32, 64, 64)
