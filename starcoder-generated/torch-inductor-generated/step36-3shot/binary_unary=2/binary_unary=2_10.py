
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(2, 6, 4, padding=1)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = torch.flatten(v1, 1)
        v3 = v2 - 52345.0
        v4 = v3 * 0.000001
        return v4
# Inputs to the model
x1 = torch.randn(1, 2, 12, 12)

