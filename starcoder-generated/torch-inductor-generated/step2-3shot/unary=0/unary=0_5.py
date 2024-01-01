
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 1, stride=1, padding=0, bias=False)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = v1 * 0.010346
        v3 = v1 + v2
        v4 = v3 * 0.496532
        v5 = torch.tanh(v4)
        v6 = v5 + 0.5
        v7 = v1 + v6
        return v7
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
