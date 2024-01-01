
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(9, 10, 1, stride=1, padding=1)
    def forward(self, x1, other):
        v1 = self.conv(x1)
        v2 = v1 + other
        return v2
# Inputs to the model
x1 = torch.randn(1, 9, 64, 64)
other = torch.randn(1, 10, 64, 64)
