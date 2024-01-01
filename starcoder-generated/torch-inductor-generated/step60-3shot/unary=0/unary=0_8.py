
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 6, 1, stride=1, padding=0)
    def forward(self, x0):
        v0 = self.conv(x0)
        v1 = v0 * 0.5
        v2 = v0 * v0
        v3 = v2 * v0
        v4 = v3 * 0.044715
        v5 = v0 + v4
        v6 = v5 * 0.7978845608028654
        v7 = torch.tanh(v6)
        v8 = v7 + 1
        v9 = v1 * v8
        return v9
# Inputs to the model
x0 = torch.randn(1, 3, 24, 32)
