
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 1, stride=1)
        self.conv2 = torch.nn.Conv2d(8, 8, 6, stride=1)
    def forward(self, x1):
        v1 = 3 + self.conv(x1)
        v2 = self.conv(x1)
        v3 = torch.relu6(v1) + v2
        v4 = v3 / 6
        return v4
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
