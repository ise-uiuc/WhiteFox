
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        v1 = self.conv2d(x1, 3, 8, 1, 1, 1)
        v2 = v1 + 3
        v3 = v2.clamp_min(0)
        v4 = v3.clamp_max(6)
        v5 = v1 * v4
        v6 = v5 / 6
        return v6
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
