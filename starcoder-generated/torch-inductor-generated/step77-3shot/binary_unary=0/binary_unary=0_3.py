
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3, groups=2)
        self.conv2 = torch.nn.Conv2d(16, 16, 5, stride=1, padding=2, groups=4)
        self.conv3 = torch.nn.Conv2d(16, 16, 3, stride=1, padding=1, groups=8)
    def forward(self, x1, x2, x3):
        v1 = self.conv1(x1)
        v2 = v1 + x2
        v3 = torch.nn.functional.relu(v2)
        v4 = self.conv2(v3)
        v5 = v4 + x3
        v6 = torch.nn.functional.relu(v5)
        v7 = self.conv3(v6)
        v8 = v7 + x1
        v9 = torch.nn.functional.relu(v8)
        return v9
# Inputs to the model
x1 = torch.randn(1, 16, 64, 64)
x2 = torch.randn(1, 16, 64, 64)
x3 = torch.randn(1, 16, 64, 64)
