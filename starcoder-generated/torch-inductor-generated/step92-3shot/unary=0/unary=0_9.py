
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(2, 30, 1, stride=3, padding=6)
    def forward(self, x13):
        v1 = self.conv(x13)
        v2 = v1 * 0.5
        v3 = v1 * v1
        v4 = v3 * v1
        v5 = v4 * 0.044715
        v6 = v1 + v5
        v7 = v6 * 0.7978845608028654
        v8 = torch.sin(v7)
        v9 = v8 + 1
        v10 = v2 * v9
        return v10
# Inputs to the model
x13 = torch.randn(1, 2, 14, 26)
