
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(4, 12, 3, stride=2, padding=1)
    def forward(self, x):
        v1 = self.conv(x)
        v2 = v1 > 0
        v3 = v1 * 0.1
        v4 = v3 > 0
        v5 = v3 * 0.2
        v6 = torch.where(v2, v1, v3)
        v7 = torch.where(v4, v6, v5)
        return v7
# Inputs to the model
x1 = torch.randn(1, 4, 32, 32)
