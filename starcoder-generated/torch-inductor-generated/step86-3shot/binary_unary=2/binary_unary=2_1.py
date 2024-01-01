
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 16, 5, stride=1, padding=2)
        self.conv2 = torch.nn.Conv2d(16, 16, 5, stride=1, padding=2)
        self.conv3 = torch.nn.Conv2d(16, 8, 3, stride=1, padding=1)
        self.conv4 = torch.nn.Conv2d(8, 8, 3, stride=1, padding=1)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = v1 + 100
        v3 = torch.relu(v2)
        v4 = torch.tanh(v3)
        v5 = self.conv2(v4)
        v6 = v5 - 100
        v7 = torch.relu(v6)
        v8 = torch.tanh(v7)
        v9 = self.conv3(v8)
        v10 = v9 + 200
        v11 = torch.relu(v10)
        v12 = torch.tanh(v11)
        v13 = self.conv4(v12)
        v14 = v13 + 300
        v15 = torch.relu(v14)
        v16 = torch.tanh(v15)
        return v16
# Inputs to the model
x1 = torch.randn(1, 3, 32, 32)
