
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(3, 16, 1, stride=1, padding=1)
    def forward(self, x1, x2, x3, x4):
        v1 = self.conv1(x1)
        v2 = self.conv1(x2)
        v3 = self.conv2(x3)
        v4 = self.conv2(x4)
        v5 = v1 + v2 + v3 + v4
        return v5
# Inputs to the model
x1 = torch.randn(1, 3, 256, 256)
x2 = torch.randn(1, 3, 256, 256)
x3 = torch.randn(1, 3, 256, 256)
x4 = torch.randn(1, 3, 256, 256)
