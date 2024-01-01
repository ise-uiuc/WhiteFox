
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 256, 1)
        self.relu6 = torch.nn.ReLU6()
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = self.relu6(v1)
        v3 = 3 + v2
        v4 = torch.clamp_min(v3, 0)
        v5 = torch.clamp_max(v4, 6)
        v6 = v1 * v5
        v7 = v6 / 6
        return v7
# Inputs to the model
x1 = torch.randn(1, 3, 112, 112)
