
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(27, 97, 1, stride=7, padding=11)
    def forward(self, x5, x6):
        v99 = x5 + x6
        v1 = self.conv(v99)
        v2 = v1 * 0.5
        v3 = v1 * v1
        v4 = v3 * v1
        v5 = v4 * 0.044715
        v6 = v1 + v5
        v7 = v6 * 0.7978845608028654
        v8 = torch.tanh(v7)
        v9 = v8 + 1
        v10 = v2 * v9
        return v10
# Inputs to the model
x5 = torch.randn(1, 27, 45, 8)
x6 = torch.randn(1, 27, 45, 8)
