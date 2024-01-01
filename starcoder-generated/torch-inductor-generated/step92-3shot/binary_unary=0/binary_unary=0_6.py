
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3, groups=2)
        self.conv2 = torch.nn.Conv2d(16, 16, 1, stride=1, padding=0)
        self.conv3 = torch.nn.Conv2d(16, 16, 1, stride=1, padding=0)
        self.conv4 = torch.nn.Conv2d(16, 16, 1, stride=1, padding=0)
        self.conv5 = torch.nn.Conv2d(16, 16, 1, stride=1, padding=0)
    def forward(self, x1, x2, x3):
        v1 = self.conv1(x2)
        v2 = self.conv2(v1)
        v3 = torch.relu(v2)
        v4 = v1 + x3
        v5 = torch.relu(v4)
        v6 = self.conv3(v5)
        v7 = v6 + v2
        v8 = torch.relu(v7)
        v9 = self.conv4(v8)
        v10 = v9 + v5
        v11 = torch.relu(v10)
        v12 = self.conv5(v11)
        v13 = v12 + x1
        v14 = torch.relu(v13)
        return v14
# Inputs to the model
x1 = torch.randn(1, 16, 64, 64)
x2 = torch.randn(1, 16, 64, 64)
x3 = torch.randn(1, 16, 64, 64)
