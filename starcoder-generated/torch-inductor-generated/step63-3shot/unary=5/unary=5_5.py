
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose1 = torch.nn.ConvTranspose2d(19, 16, 5, stride=3, padding=1, groups=2)
    def forward(self, x1, x2):
        v1 = v5 = self.conv_transpose1(x1)
        v2 = v1 * 0.5
        v3 = v1 * 0.7071067811865476
        v4 = torch.erf(v3)
        v6 = v4 + 1
        v7 = v2 * v6
        v8 = v3 + v7
        v9 = x2 / v4
        v10 = v7 * v9
        v11 = torch.relu(v10)
        v12 = v7 - v8
        v13 = v10 - v6
        v14 = v6 * v8
        v15 = torch.ceil(v14)
        v16 = v13 + v7
        v17 = v10 + v5
        v18 = v15 * v9
        v19 = torch.tanh(v18)
        return v7, v12, v15, v16, v17, v19, v9
# Inputs to the model
x1 = torch.randn(1, 19, 64, 64)
x2 = torch.randn(1, 19, 64, 64)
