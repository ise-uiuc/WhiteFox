
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(8, 8, 1, stride=1, padding=1)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = v1 + 0.7978845608028654
        v3 = torch.tanh(v2)
        v4 = v3 + 1
        v5 = v1 * v4
        return v5
# Inputs to the model
x1 = torch.randn(1, 8, 32, 32)
