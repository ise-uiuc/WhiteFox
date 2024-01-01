
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3)
        self.conv2 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3)
        self.conv3 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3)
    def forward(self, x1, x2, x2_2):
        v1 = self.conv1(x1)
        v2 = self.conv1(x1)
        a3 = self.conv3(x1)
        v3 = v1 + x2
        v4 = self.conv1(x1)
        v5 = v2 + v3
        v6 = torch.relu(v5)
        v7 = a3 + x2_2
        v8 = self.conv2(v5)
        v9 = v8 + v7
        v10 = torch.relu(v9)
        v11 = v10 + x2_2
        v12 = self.conv3(v11)
        v13 = v12 + v11
        v14 = torch.relu(v13)
        return v14
# Inputs to the model
x1 = torch.randn(1, 16, 64, 64)
x2 = torch.randn(1, 16, 64, 64)
x2_2 = torch.randn(1, 16, 64, 64)
