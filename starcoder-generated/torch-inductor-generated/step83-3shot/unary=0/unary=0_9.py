
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(4, 690, 2, stride=43, padding=3)
    def forward(self, x0a):
        v1 = self.conv(x0a)
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
x0a = torch.randn(1, 4, 73, 18)
