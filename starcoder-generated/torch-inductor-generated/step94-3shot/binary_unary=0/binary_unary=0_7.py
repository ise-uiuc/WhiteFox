
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(16, 16, 3, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(16, 16, 3, stride=1, padding=1)
    def forward(self, x1, x2, x3, x4, x5, x6):
        v1 = self.conv1(x1)
        v2 = x2 + v1
        v3 = torch.relu(v2)
        v4 = self.conv2(v3)
        v5 = v2 + v4
        v6 = torch.relu(v5)
        v7 = v5 + x6
        v8 = torch.relu(v7)
        return v8
# Inputs to the model
x1 = torch.randn(1, 16, 128, 128)
x2 = torch.randn(1, 16, 128, 128)
x3 = torch.randn(1, 16, 128, 128)
x4 = torch.randn(1, 16, 128, 128)
x5 = torch.randn(1, 16, 128, 128)
x6 = torch.randn(1, 16, 128, 128)
