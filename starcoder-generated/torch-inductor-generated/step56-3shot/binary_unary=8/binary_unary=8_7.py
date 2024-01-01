
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(12, 25, 3, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(12, 25, 3, stride=1, padding=1)
        self.conv3 = torch.nn.Conv2d(16, 25, 3, stride=1, padding=1)
        self.conv4 = torch.nn.Conv2d(16, 25, 3, stride=1, padding=1)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = self.conv2(x1)
        v3 = self.conv1(x1)
        v4 = self.conv2(x1)
        v5 = v1 + v2
        v6 = torch.relu(v5)
        v7 = v3 + v4
        v8 = v7+v7+v7+v7+v7+v7+v7+v7+v7+v7+v7
        v9 = torch.relu(v8)
        v10 = self.conv3(torch.relu(v9))
        v11 = self.conv4(torch.relu(v10))
        v12 = v11 + v11 + v11 + v11 + v11 + v11 + v11 + v11
        v13 = torch.relu(v12)
        return v13
# Inputs to the model
x1 = torch.randn(1, 12, 64, 64)
