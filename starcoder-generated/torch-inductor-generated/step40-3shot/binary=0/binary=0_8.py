
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
    def forward(self, x1, other=1, padding2=None, padding3=None):
        v1 = self.conv(x1)
        v2 = v1 + other
        v3 = v2 - padding1
        v4 = v3 + padding2
        return v4
# Inputs to the model
x1 = torch.randn(1, 3, 32, 32)
