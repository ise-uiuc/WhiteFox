
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_1 = torch.nn.Conv2d(3, 8, 3, stride=1, padding=1)
    def forward(self, x1):
        v1 = self.conv_1(x1)
        v2 = v1 - 2
        v3 = torch.clamp_min(v2, 0)
        v4 = torch.clamp_max(v3, 6)
        v5 = v1 * v4
        v6 = v5 / 6
        return v6
# Inputs to the model
x1 = torch.randn(1, 3, 256, 256)
