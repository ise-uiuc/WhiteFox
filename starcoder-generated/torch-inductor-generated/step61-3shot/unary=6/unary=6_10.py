
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 32, 1, stride=1, padding=1)
        self.prelu = torch.nn.PReLU()
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = v1 + 3
        v3 = torch.clamp(v2, 0, 6)
        v4 = v1 * v3
        v5 = v4 / 6
        v6 = self.prelu(v5)
        return v6 + v5
# Inputs to the model
x1 = torch.randn(1, 3, 128, 128)
