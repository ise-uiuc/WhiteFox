
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 16, 3, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(16, 8, 1, stride=1, padding=1)
        self.conv3 = torch.nn.Conv2d(16, 4, 1, stride=1, padding=1)
        self.conv4 = torch.nn.Conv2d(16, 4, 1, stride=1, padding=1)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = self.conv2(x1)
        v3 = torch.relu(v1)
        v4 = v3 + v2
        v5 = self.conv3(v4)
        v6 = torch.relu(v5)
        v7 = v6 + v4
        v8 = self.conv4(v7)
        v9 = torch.relu(v8)
        return v9
# Inputs to the model
x1 = torch.randn(1, 3, 30, 30)
