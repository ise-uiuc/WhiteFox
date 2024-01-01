
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3)
        self.conv2 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3)
        self.conv3 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3)
        self.conv4 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3)
    def forward(self, x):
        v0 = self.conv1(x)
        v1 = self.conv2(x)
        v2 = v0 + v1
        v3 = torch.relu(v2)
        v4 = torch.relu(v0)
        v5 = self.conv3(v4)
        v6 = v5 + v1
        v7 = torch.relu(v6)
        v8 = torch.relu(v1)
        v9 = self.conv4(v8)
        v10 = v9 + v0
        v11 = torch.relu(v10)
        return v11
# Inputs to the model
x = torch.randn(1, 16, 64, 64)
