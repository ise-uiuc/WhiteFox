
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 4, 1, stride=1, padding=1)
        self.relu = torch.nn.ReLU()
        self.other_conv = torch.nn.Conv2d(4, 8, 1, stride=1, padding=1)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = self.relu(v1)
        v3 = self.other_conv(v2)
        t0 = 0.25
        v4 = t0 * v3
        t1 = 3
        v5 = v4 + t1
        t2 = v5.clamp_min(0)
        t3 = t2.clamp_max(6)
        t4 = t3 / 6
        return t4
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
