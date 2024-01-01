
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(3, 4, 1, stride=1, padding=1)
    def forward(self, x1):
        v1 = self.conv(x1) * 12
        v2 = self.conv2(x1) * 1
        v3 = v1 + v2
        v4 = v2 + v3
        return v4
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
