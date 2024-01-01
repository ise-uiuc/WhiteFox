
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = torch.nn.Conv2d(16, 16, 7, stride=2, padding=3)
        self.conv2 = torch.nn.Conv2d(16, 16, 3, stride=1, padding=1)
        self.conv3 = torch.nn.Conv2d(16, 16, 3, stride=2, padding=1)
    def forward(self, x1, x2, x3):
        v1 = self.conv1(x1)
        v2 = torch.relu(x1)
        v3 = v2 + x2
        v4 = v3 + x3
        v5 = torch.relu(v1)
        v6 = v5 + v4
        v7 = torch.relu(self.conv2(v6))
        v8 = torch.relu(x3)
        v9 = self.conv3(x3)
        v10 = v7 + v8
        v11 = torch.relu(v9)
        return v10
# Inputs to the model:
x1 = torch.randn(1, 16, 64, 64)
x2 = torch.randn(1, 16, 64, 64)
x3 = torch.randn(1, 16, 64, 64)
