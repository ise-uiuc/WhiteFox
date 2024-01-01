


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 4, 3, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(4, 4, 3, stride=1, padding=1)
    def forward(self, x):
        v1 = self.conv1(x)
        v2 = self.conv2(v1)
        v3 = v2 + 3
        v4 = torch.clamp(v3, 0, 6)
        v5 = v1 * v4
        v6 = v5 / 6
        v7 = self.conv2(v6)
        return v7
# Inputs to the model
x = torch.randn(1, 3, 32, 32)
