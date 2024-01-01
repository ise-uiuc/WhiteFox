
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3, groups=8)
        self.conv2 = torch.nn.Conv2d(32, 16, 7, stride=1, padding=3)
        self.conv3 = torch.nn.Conv2d(16, 32, 7, stride=1, padding=3)
    def forward(self, x1):
        v4 = self.conv1(x1)
        v1 = self.conv1(v4)
        v2 = v1 + v4
        v3 = torch.nn.functional.relu(v2)
        v5 = self.conv2(v3)
        v6 = v5 + x1
        v7 = torch.nn.functional.relu(v6)
        v8 = self.conv3(v7)
        v9 = v8 + v5
        v10 = torch.nn.functional.relu(v9)
        return v10
# Inputs to the model
x1 = torch.randn(2, 16, 64, 64)
