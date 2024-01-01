
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 1, 1, stride=1, padding=0)
        self.conv2 = torch.nn.Conv2d(1, 1, 1, stride=1, padding=0)
        self.pool = torch.nn.MaxPool2d(3, stride=1, padding=1)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = 3 + v1
        v3 = torch.clamp_min(v2, 0)
        v4 = torch.clamp_max(v3, 6)
        v5 = v1 * v4
        v6 = v5 / 6
        v7 = self.pool(v6)
        v8 = self.conv2(v7)
        v9 = v8 * 3 + 0.1
        return v9
# Inputs to the model
x1 = torch.randn(2, 3, 64, 64)
