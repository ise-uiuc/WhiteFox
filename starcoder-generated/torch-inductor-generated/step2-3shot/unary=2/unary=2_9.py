
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1d = torch.nn.ConvTranspose1d(219, 92, 1, stride=1, padding=1, groups=2)
    def forward(self, x7):
        v1 = self.conv1d(x7)
        v2 = v1 * 0.5
        v3 = v1 * v1 * v1
        v4 = v3 * 0.044715
        v5 = v1 + v4
        v6 = v5 * 0.7978845608028654
        v7 = torch.tanh(v6)
        v8 = v7 + 1
        v9 = v2 * v8
        return v9
# Inputs to the model
x7 = torch.randn(1, 219, 208)
