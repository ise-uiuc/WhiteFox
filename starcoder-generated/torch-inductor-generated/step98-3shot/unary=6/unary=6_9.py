
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 15, 1, stride=1, padding=1)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = 3.3 + v1
        v3 = torch.clamp_min(v2, 0)
        v4 = torch.relu6(2*v3)
        v5 = v1 * v4
        v6 = v5 / 23
        return v6
# Inputs to the model
x1 = torch.randn(1, 3, 16, 16)
