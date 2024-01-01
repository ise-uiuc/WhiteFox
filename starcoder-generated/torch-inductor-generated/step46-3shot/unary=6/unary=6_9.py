
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(3, 4, 1, stride=1, padding=1)
        self.conv3 = torch.nn.Conv2d(4, 6, 1, stride=1, padding=1)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = self.conv2(x1)
        v3 = v1 + v2
        v4 = self.conv3(v3)
        v5 = v4 + 3
        v6 = torch.clamp_min(v5, 0)
        v7 = torch.clamp_max(v6, 6)
        v8 = v4 * v7
        v9 = v8 / 6
        return v9
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
