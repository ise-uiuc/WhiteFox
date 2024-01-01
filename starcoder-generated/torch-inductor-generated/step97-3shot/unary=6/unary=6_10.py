
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 3, 3, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(3, 3, 3, stride=1, padding=1)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = self.conv2(x1)
        v3 = 3 + v2 + v2
        v4 = v2 + 6
        v5 = v2 * v4
        v6 = v2 / v4
        return v1 + v3 + v5 + v6
# Inputs to the model
x1 = torch.randn(1, 3, 32, 32)
