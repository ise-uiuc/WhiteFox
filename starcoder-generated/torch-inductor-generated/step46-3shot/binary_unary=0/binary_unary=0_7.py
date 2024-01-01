
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3)
        self.conv2 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3)
        self.conv3 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3)
    def forward(self, x):
        v1 = self.conv1(x)
        v2 = torch.ones_like(v1)
        v4 = v1 + v2
        v5 = torch.relu(v4)
        v7 = (v5 ** 4)
        v8 = self.conv2(v7)
        v9 = (v1 + x)
        v10 = torch.relu(v9)
        v11 = (v7 + v8)
        v12 = torch.relu(v11)
        v13 = self.conv3(v12)
        v14 = torch.ones_like(v13)
        v16 = v13 + v14
        v17 = torch.relu(v16)
        return v17
# Inputs to the model
x = torch.randn(1, 16, 64, 64)
