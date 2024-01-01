
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3)
        self.conv2 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3)
        self.conv3 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3)
        self.conv4 = torch.nn.Conv2d(16, 16, 1, stride=1, padding=0)
    def forward(self, x1, x2, x3):
        v1 = self.conv1(x1)
        v2 = v1 + x1
        v3 = self.conv2(v2)
        v4 = v3 + x2
        v5 = self.conv3(v4)
        v6 = torch.relu(v5)
        v7 = self.conv4(v6)
        v8 = x1 + v7
        v9 = v8 + x3
        v10 = torch.relu(v9)
        return v10
# Inputs to the model
x1 = torch.randn(1, 16, 64, 64)
x2 = torch.randn(1, 16, 64, 64)
x3 = torch.randn(1, 64, 64, 1)
