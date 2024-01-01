
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 3, 1, stride=1, padding=1)
        self.clamp_min_0 = torch.nn.ReLU6(inplace=True)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = torch.clamp_min(v1, 0)
        v3 = v1 * v2
        v4 = v3 / 6
        return v4
# Inputs to the model
x1 = torch.randn(3, 8, 64, 64)
