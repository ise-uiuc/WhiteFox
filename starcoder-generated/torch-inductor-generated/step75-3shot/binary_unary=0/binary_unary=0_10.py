
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(16, 32, 15, stride=1, padding=3)
        self.conv2 = torch.nn.Conv2d(32, 32, 11, stride=1, padding=3)
        self.conv3 = torch.nn.Conv2d(32, 16, 7, stride=1, padding=3)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = torch.nn.BatchNorm2d(v1)
        v3 = torch.nn.ReLU()(v2)
        v4 = v3 + x1
        v5 = torch.nn.ReLU()(v4)
        v6 = self.conv2(v5)
        v7 = self.conv3(v5)
        v8 = v6 + v7
        v9 = torch.nn.ReLU()(v8)
        return v9
# Inputs to the model
x1 = torch.randn(1, 16, 32, 32)
