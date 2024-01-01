
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 16, 1, stride=1, padding=2)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = v1 + 3
        v3 = torch.floor(v2)
        v4 = v1 * v3
        v5 = torch.round(v4)
        v6 = v5 - 6
        return v6
# Inputs to the model
x1 = torch.randn(10, 3, 100, 100)
