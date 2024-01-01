
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(4, 4, groups=2)
    def forward(self, x1, x2):
        v1 = self.conv_transpose(x1)
        v2 = v1 * 0.5
        v3 = v1 * v1 * v1
        v4 = v3 * 0.044715
        v5 = v1 +v4
        v6 = v5 * 0.7978845608028654
        v7 = torch.tanh(v6)
        v8 = v7 + 1
        v9 = v2 * v8
        v10 = self.conv_transpose(x2)
        v11 = v10 * 0.5
        v12 = v10 * v10 * v10
        v13 = v12 * 0.044715
        v14 = v10 + v13
        v15 = v14 * 0.7978845608028654
        v16 = torch.tanh(v15)
        v17 = v16 + 1
        v18 = v11 * v17
        return v9 + v18
# Inputs to the model
x1 = torch.randn(3, 4, 7, 7)
x2 = torch.randn(3, 4, 9, 9)
