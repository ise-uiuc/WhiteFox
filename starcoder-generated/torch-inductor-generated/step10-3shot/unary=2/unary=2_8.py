
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(3, 16, 1, stride=1, padding=1)
    def forward(self, x1):
        v1 = self.conv_transpose(x1)
        v2 = v1 * 0.5
        v3 = v1 * v1 * v1
        v4 = v3 * 0.044715
        v5 = v1 + v4
        v6 = v5 * 0.7978845608028654
        v7 = v6 - v2
        v8 = torch.tanh(v7) * v6
        v9 = v8 + v6 * 0.7978845608028654 + 1
        return v9
# Inputs to the model
x1 = torch.randn(1, 3, 3, 3)
