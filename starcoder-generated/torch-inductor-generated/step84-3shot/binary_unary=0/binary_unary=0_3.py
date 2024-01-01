
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3)
        self.conv2 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3)
        self.conv3 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3)
    def forward(self, x1, x2, x3, x4):
        v1 = self.conv1(x1)
        v2 = v1 + x2
        v3 = self.conv2(x3)
        v4 = v3 + x4
        v5 = v2 + v4
        v6 = torch.nn.ReLU()(v5)
        v7 = self.conv3(v6)
        v8 = x1 + v7
        v9 = torch.nn.ReLU()(v8)
        return v9
# Inputs to the model
x1 = torch.randn(1, 16, 64, 64)
x2 = torch.randn(1, 16, 64, 64)
x3 = torch.randn(1, 16, 64, 64)
x4 = torch.randn(1, 16, 64, 64)
