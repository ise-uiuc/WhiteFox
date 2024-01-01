
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3)
        self.conv2 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3)
        self.conv3 = torch.nn.Conv2d(16, 16, 1, stride=1, padding=0)
    def forward(self, x1, x2, x3):
        v1 = self.conv1(x1)
        a1 = self.conv3(x1)
        v2 = v1 + x2
        v3 = torch.relu(v2)
        v4 = a1 + x2
        v5 = torch.relu(v4)
        v6 = self.conv2(v5)
        v7 = v6 + x3
        v8 = torch.relu(v7)
        return v8
# Inputs to the model
x1 = torch.randn(1, 16, 64, 64)
x2 = torch.randn(1, 16, 64, 64)
x3 = torch.randn(1, 16, 64, 64)
