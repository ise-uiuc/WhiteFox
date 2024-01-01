
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(16, 2048, 5, stride=1, padding=0)
        self.conv_transpose_1 = torch.nn.ConvTranspose2d(16, 2048, 3, stride=1, padding=0)
    def forward(self, x1):
        v1 = self.conv_transpose(x1)
        v2 = v1 * 0.5
        v3 = v1 * 0.7071067811865476
        v4 = torch.erf(v3)
        v5 = v4 + 1
        v6 = v2 * v5
        v7 = self.conv_transpose_1(x1)
        v8 = v6 * 0.5
        v9 = v6 * 0.7071067811865476
        v10 = torch.erf(v9)
        v11 = v10 + 1
        v12 = v8 * v11
        return v12
# Inputs to the model
x1 = torch.randn(1, 16, 69, 63)
