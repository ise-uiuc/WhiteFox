
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 32, 3, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(32, 64, 3, stride=1, padding=1)
    def forward(self, x1):
        v1 = 3 + x1
        v2 = self.conv1(v1)
        t1 = self.conv2(v2)
        v3 = 3 + t1
        v4 = torch.clamp_min(v3, 0)
        v5 = torch.clamp_max(v4, 6)
        v6 = v2 * v5
        v7 = v6 / 6
        v8 = v1 * v7
        v9 = v8 / 6
        return v9
# Inputs to the model
x1 = torch.randn(2, 3, 32, 32)
