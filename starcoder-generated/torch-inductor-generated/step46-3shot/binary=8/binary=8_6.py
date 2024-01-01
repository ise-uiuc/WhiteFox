
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(32, 256, 1, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(256, 32, 1, stride=1, padding=1)
    def forward(self, x1, x2):
        v1 = self.conv1(x1)
        v2 = self.conv2(x2)
        v3 = v1 + v2
        v4 = v1 - x2
        v5 = v2 * x1
        return v3, v4, v5
# Inputs to the model
x1 = torch.randn(1, 32, 64, 64)
x2 = torch.randn(1, 256, 64, 64)
