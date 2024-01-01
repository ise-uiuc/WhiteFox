
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(3, 8, 1, stride=1, padding=0)
    def forward(self, x1):
        v1 = self.conv_transpose(x1)
        v2 = 0.5 * v1
        v3 = v1 * v1 * v1
        v4 = 0.044715 * v3
        v5 = v1 + v4
        v6 = 0.7978845608028654 * v5
        v7 = torch.tanh(v6)
        v8 = v2 + v1
        v9 = 1 + v7
        v10 = v9 * v9
        v11 = v10 * v3
        return v1
# Inputs to the model
x1 = torch.randn(3, 3, 32, 32)
