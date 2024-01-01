
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(20, 8, 1, stride=1, padding=0)
    def forward(self, x1, other):
        v1 = self.conv(x1)
        v2 = v1 + other
        v3 = torch.cat([v2, other, other])
        return v3
# Inputs to the model
x1 = torch.randn(1, 20, 64, 64)
other = 1
