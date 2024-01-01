
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3)
        self.conv2 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3)
        self.conv3 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3)
    def forward(self, x1, x2, x3):
        v1 = self.conv1(x1)
        v2 = self.conv1(x2)
        v3 = self.conv1(x3)
        v4 = v1 + v2
        v5 = torch.relu(v4)
        v6 = v1 + v3
        v7 = torch.relu(v6)
        v8 = self.conv2(v5)
        v9 = self.conv3(v7)
        v10 = v8 + v9
        v11 = torch.nn.functional.relu(v10)
        return v11
# Inputs to the model
x1 = torch.randn(1, 16, 64, 64)
x2 = torch.randn(1, 16, 64, 64)
x3 = torch.randn(1, 16, 64, 64)
