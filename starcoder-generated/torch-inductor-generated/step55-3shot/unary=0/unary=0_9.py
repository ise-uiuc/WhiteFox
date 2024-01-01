
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 3, 5, stride=2, padding=2)
        self.conv2 = torch.nn.Conv2d(200, 100, 2, stride=1, padding=1)
        self.conv3 = torch.nn.Conv2d(123, 123, 2, stride=1, padding=1)
    def forward(self, x4, x3, x1):
        v0 = torch.add(x3, x1)
        v1 = self.conv1(x4)
        v2 = self.conv2(v0)
        v3 = v2 + 1
        v4 = torch.mul(v1, v3)
        v5 = v4 + 0.01
        v6 = self.conv3(v5)
        return v4
# Inputs to the model
x4 = torch.randn(1, 1, 64, 64)
x3 = torch.randn(4, 200, 8, 4)
x1 = torch.randn(2, 123, 32, 32)
