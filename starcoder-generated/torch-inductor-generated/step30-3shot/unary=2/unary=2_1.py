
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose1 = torch.nn.ConvTranspose2d(3, 5, 3, padding=2)
        self.conv_transpose2 = torch.nn.ConvTranspose2d(5, 1, 2, padding=(8, 16))
    def forward(self, x1):
        v1 = self.conv_transpose1(x1)
        v2 = v1 * 0.5
        v3 = v1 * v1 * v1
        v4 = v3 * 0.044715
        v5 = v1 + v4
        v6 = v5 * 0.7978845608028654
        v7 = torch.tanh(v6)
        v8 = v7 + 1
        v9 = v2 * v8
        v10 = self.conv_transpose2(v9)
        v11 = v10 * 0.1
        v12 = v9 * 0.01
        v13 = v10 + v11
        v14 = v11 + v12
        v15 = v13 * v14
        return v15
# Inputs to the model
x1 = torch.randn(1, 3, 122, 4)
