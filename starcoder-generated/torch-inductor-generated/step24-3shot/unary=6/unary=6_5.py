
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 8, 1, stride=1, padding=3)
        self.conv2 = torch.nn.Conv2d(8, 24, 3, stride=1, padding=3)
        self.conv3 = torch.nn.Conv2d(24, 16, 1, stride=1, padding=3)
    def forward(self, x):
        v1 = self.conv1(x)
        v2 = self.conv2(v1)
        v3 = self.conv3(v2)
        v4 = v3 + 3
        v5 = torch.clamp_min(v4, 0)
        v6 = torch.clamp_max(v5, 6)
        v7 = v3 * v6
        v8 = v7 / 6
        return v8
# Inputs to the model
x = torch.randn(1, 3, 256, 256)
