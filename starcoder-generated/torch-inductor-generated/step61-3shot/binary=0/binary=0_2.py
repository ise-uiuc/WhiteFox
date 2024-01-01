
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(25, 11, 1, stride=1, padding=1)
    def forward(self, x1, other):
        v1 = self.conv(x1 * other)
        v2 = v1 * other
        return v2
# Inputs to the model
x1 = torch.randn(1, 25, 64, 64)
