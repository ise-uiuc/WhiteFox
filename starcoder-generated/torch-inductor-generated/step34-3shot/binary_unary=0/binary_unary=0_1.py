
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(16, 16, 7, stride=2, padding=3)
        self.conv2 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3)
        self.conv3 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3)
        self.conv4 = torch.nn.Conv2d(16, 16, 5, stride=1, padding=1)
    def forward(self, x1, x2, x3):
        v1 = self.conv1(x1)
        v2 = v1
        v3 = v2
        v4 = v3 + x2
        v5 = torch.relu(v4)
        v6 = self.conv2(v5)
        v7 = v6 + v5
        v8 = torch.relu(v7)
        v9 = self.conv3(v8)
        v10 = v9 + x3
        v11 = torch.relu(v10)
        v12 = self.conv4(v11)
        return v12
# Inputs to the model
x1 = torch.randn(1, 16, 64, 64)
x2 = torch.randn(1, 16, 64, 64)
x3 = torch.randn(1, 16, 64, 64)
