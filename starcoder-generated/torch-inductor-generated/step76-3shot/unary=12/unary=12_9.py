
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 3, 3, stride=2, padding=1)
        self.conv1 = torch.nn.Conv2d(3, 3, 5, stride=2, padding=2)
        self.conv2 = torch.nn.Conv2d(3, 1, 7, stride=3, padding=3)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = self.conv1(v1)
        v3 = self.conv2(v2)
        v4 = v2 * v3
        return v4
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
