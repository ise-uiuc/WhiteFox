
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(4, 1, 1, stride=1, padding=0)
        self.conv2 = torch.nn.Conv2d(1, 4, 1, stride=1, padding=0)
    def forward(self, x1):
        v1 = self.conv(x1)
        v3 = v1 * 0.7071067811865476
        v4 = torch.erf(v3)
        v6 = v4 + 10
        v7 = self.conv2(v6)
        return v7
# Inputs to the model
x1 = torch.randn(1, 4, 32, 32)
