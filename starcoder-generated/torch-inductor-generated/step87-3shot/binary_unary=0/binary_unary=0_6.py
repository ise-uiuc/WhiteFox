
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3)
        self.conv2 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3)
        self.conv3 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3)
        self.conv4 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3)
    def forward(self, x):
        v1 = self.conv1(x)
        v2 = self.conv2(x)
        v3 = self.conv3(x)
        v4 = self.conv4(x)
        v5 = v1 + v2 + v3 + v4
        v6 = torch.relu(v5)
        v7 = v4 + v6
        v8 = self.conv1(v7)
        v9 = self.conv2(v7)
        v10 = self.conv3(v7)
        v11 = v8 + v9 + v10 + v8 + v10
        v12 = torch.relu(v11)
        v13 = v12 + v10
        v14 = torch.relu(v13)
        v15 = v14 + v12
        v16 = torch.relu(v15)
        return v16
# Inputs to the model
x = torch.randn(1, 16, 64, 64)
