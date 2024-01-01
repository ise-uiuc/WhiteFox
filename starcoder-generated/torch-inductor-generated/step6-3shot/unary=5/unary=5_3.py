
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose1d(8, 8, 3, stride=2, padding=1)
    def forward(self, x1):
        v1 = self.conv_transpose(x1)
        v2 = v1 * 3.0E-05
        v3 = v1 * 1.52587890625E-08
        v4 = torch.erf(v3)
        v5 = v4 + 1
        v6 = v2 * v5
        v7 = torch.cat([v6, v6, v6], 1)
        v8 = self.conv_transpose(v7)
        v9 = v8 * 3.0E-05
        v10 = v8 * 1.52587890625E-08
        v11 = torch.erf(v10)
        v12 = v11 + 1
        v13 = v9 * v12
        v14 = torch.cat([v6, v13], 1)
        v15 = self.conv_transpose(v14)
        v16 = v15 * 3.0E-05
        v17 = v15 * 1.52587890625E-08
        v18 = torch.erf(v17)
        v19 = v18 + 1
        v20 = v16 * v19
        return v20
# Inputs to the model
x1 = torch.randn(1, 8, 64)
