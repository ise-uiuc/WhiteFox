
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(32, 32, (1, 1), stride=1, padding=0, groups=16)
        self.conv2 = torch.nn.Conv2d(32, 16, (1, 1), stride=1, padding=0)
        self.conv3 = torch.nn.Conv2d(16, 16, (1, 1), stride=1, padding=0)
    def forward(self, x1, x2, x3):
        v1 = self.conv1(x1)
        v2 = self.conv2(v1)
        v3 = torch.relu(v2)
        v4 = self.conv3(v3)
        v5 = x2 - v4
        v6 = torch.relu(v5)
        v7 = v6 + x1
        return v7
# Inputs to the model
x1 = torch.randn(1, 32, 128, 64)
x2 = torch.randn(1, 32, 84, 128)
x3 = torch.randn(1, 32, 84, 128)
