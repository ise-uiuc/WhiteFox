
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(3, 3, kernel_size=1, stride=1, padding=0)
    def forward(self, x1, x2):
        v1 = self.conv_transpose(x1)
        v2 = v1 * 0.5
        v3 = x2 * v1 * v1
        v4 = v3 * 0.044715
        v5 = v1 + v4
        v6 = v1 + v3
        v7 = torch.sin(v6)
        v8 = v5 + v7
        v9 = v8 * 0.7978845608028654
        v10 = torch.tanh(v9)
        v11 = v10 + 1
        v12 = v2 * v1 + v3 * v8
        v13 = v12 * 1.4142135623730951
        return v13
# Inputs to the model
x1 = torch.randn(1, 3, 3, 3)
x2 = torch.randn(2, 3, 2, 2)
