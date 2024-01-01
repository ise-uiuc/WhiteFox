
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(8, 1, kernel_size=1, stride=2, padding=0)
        self.conv_transpose2 = torch.nn.ConvTranspose2d(8, 2, kernel_size=1, stride=2, padding=0)
    def forward(self, x1, x2):
        v1 = self.conv_transpose(x1)
        v2 = v1 * 0.5
        v3 = v1 * v1 * v1
        v4 = v3 * 0.044715
        v5 = v1 + v4
        v6 = v5 * 0.7978845608028654
        v7 = torch.tanh(v6)
        v8 = v7 + 1
        v9 = v2 * v8
        v10 = v1 + v9
        v11 = torch.tanh(v10)
        v12 = self.conv_transpose2(v11) * 0.5
        v13 = v12 * v12 * v12
        v14 = v13 + 0.044715
        v15 = v12 + v14
        v16 = v15 * 0.7978845608028654
        v17 = x2 + v16
        return v17

# Inputs to the model
x1 = torch.randn(1, 8, 4, 4)
x2 = torch.randn(1, 8, 4, 4)
