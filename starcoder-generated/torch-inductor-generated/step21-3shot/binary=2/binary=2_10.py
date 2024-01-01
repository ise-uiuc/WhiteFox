
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(5, 4, 1, stride=1, padding=1)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = torch.cat([v1, v1, v1], dim=1)
        v3 = v2 - 0.15
        return v3
# Inputs to the model
x1 = torch.randn(1, 5, 64, 64)
