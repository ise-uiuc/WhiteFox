
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(5, 8, 1, stride=1, padding=0)
        self.gelu = torch.nn.GELU()
    def forward(self, x5):
        v1 = self.conv(x5)
        v3 = self.gelu(v1)
        v4 = v3 * 0.5
        v6 = v3 * v3
        v7 = v6 * v3
        v8 = v7 * 0.044715
        v9 = v3 + v8
        v10 = v9 * 0.7978845608028654
        v11 = self.gelu(v10)
        v12 = v11 + 1
        v13 = v4 * v12
        return v13
# Inputs to the model
x5 = torch.randn(1, 5, 30, 30)
