
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(4, 1, 3, stride=2, padding=0)
    def forward(self, x):
        v1 = self.conv(x)
        v2 = v1 > 0
        v3 = v1 * -0.92
        v4 = torch.where(v2, v1, v3)
        return v4
# Inputs to the model
x1 = torch.randn(1, 16, 64, 16)
