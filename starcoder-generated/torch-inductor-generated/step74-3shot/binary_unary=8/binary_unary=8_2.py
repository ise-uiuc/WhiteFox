
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 16, 3, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(3, 8, 3, stride=1, padding=1)
        self.conv3 = torch.nn.Conv2d(8, 4, 1, stride=1, padding=0)
        self.conv4 = torch.nn.Conv2d(8, 8, 1, stride=2)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = self.conv2(x1)
        v3 = self.conv3(v2)
        v4 = self.conv3(v2)
        v5 = self.conv3(v2)
        v6 = v3 + v4 + v5
        v7 = torch.relu(v6)
        v8 = torch.relu(v6)
        v9 = self.conv4(v7)
        v10 = torch.relu(v9)
        v11 = self.conv4(v8)
        v12 = torch.relu(v11)
        v13 = v10 + v12
        return v13
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
