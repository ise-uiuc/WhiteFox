
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = torch.nn.MaxPool2d(16, stride=16)
        self.conv = torch.nn.Conv2d(6, 6, 1)
    def forward(self, x1):
        v1 = self.pool(x1)
        v2 = 4 + v1
        v3 = torch.clamp_min(v2, 0)
        v4 = torch.clamp_max(v3, 6)
        v5 = self.conv(v4)
        v6 = torch.clamp_min(v5, 0)
        v7 = torch.clamp_max(v6, 6)
        v8 = v5 * v7
        v9 = v8 / 6
        return v9
# Inputs to the model
x1 = torch.randn(1, 6, 256, 256)
