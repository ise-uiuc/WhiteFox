
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3)
        self.conv2 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3)
        self.conv3 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3)
    def forward(self, x1, x2, x3, x4, x5):
        v1 = self.conv1(x1)
        v2 = v1 + x2
        v3 = self.conv2(x2)
        v4 = torch.nn.ReLU()(v2)
        v5 = v3 + x3
        v6 = torch.nn.ReLU()(v5)
        v7 = v4 + v6
        v8 = torch.nn.ReLU()(v7)
        v9 = self.conv3(v8)
        v10 = v9 + x4
        v11 = torch.nn.ReLU()(v10)
        v12 = v11 + x5
        v13 = torch.nn.ReLU()(v12)
        return v13
# Inputs to the model
x1 = torch.randn(1, 16, 64, 64)
x2 = torch.randn(1, 16, 64, 64)
x3 = torch.randn(1, 16, 64, 64)
x4 = torch.randn(1, 16, 64, 64)
x5 = torch.randn(1, 16, 64, 64)
