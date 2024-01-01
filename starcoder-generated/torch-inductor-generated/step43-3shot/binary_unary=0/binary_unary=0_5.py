
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3)
        self.conv2 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3)
    def forward(self, x1, x2):
        v1 = self.conv1(x1)
        v2 = torch.nn.functional.pad(v1, [1, 1, 1, 1])
        v3 = self.conv2(v2)
        v12 = v3 + x2
        v13 = torch.relu(v12)
        v14 = self.conv1(v13)
        v4 = v14 + v12
        v5 = torch.relu(v4)
        v6 = self.conv2(v5)
        v7 = v6 + v3
        v8 = torch.relu(v7)
        return v8
# Inputs to the model
x1 = torch.randn(1, 16, 64, 64)
x2 = torch.randn(1, 16, 64, 64)
