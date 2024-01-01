
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(16, 32, 3, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(32, 32, 16, stride=1, padding=0)
        self.conv3 = torch.nn.Conv2d(32, 64, 1, stride=1, padding=0)
        self.conv4 = torch.nn.Conv2d(64, 128, 1, stride=1, padding=0)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = v1 - 100
        v3 = F.relu(v2)
        v4 = self.conv2(v3)
        v5 = v4 - 200
        v6 = F.relu(v5)
        v7 = self.conv3(v6)
        v8 = v7 - 300
        v9 = F.relu(v8)
        v10 = self.conv4(v9)
        v11 = v10 - 400
        v12 = F.relu(v11)
        return v12

# Inputs to the model
x1 = torch.randn(1, 16, 56, 56)

