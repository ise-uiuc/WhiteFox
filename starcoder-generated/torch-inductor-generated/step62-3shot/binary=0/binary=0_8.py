
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(6, 1, 1, stride=1, padding=1)
    def forward(self, x1, other):
        v1 = self.conv(x1)
        v2 = v1 + other
        v3 = v2 + other
        return v3
# Inputs to the model
x1 = torch.randn(1, 6, 64, 64)
