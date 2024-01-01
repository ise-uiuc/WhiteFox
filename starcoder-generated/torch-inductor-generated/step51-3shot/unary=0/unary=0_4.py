
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(6, 16, 6, stride=3, padding=2)
        self.conv_1 = torch.nn.ConvTranspose2d(16, 8, 2, stride=2)
        self.conv_2 = torch.nn.ConvTranspose2d(16, 8, 4, stride=2)
    def forward(self, x0):
        v1 = self.conv(x0)
        v2 = v1 * 0.5
        v3 = v1 * v1
        v4 = v3 * v1
        v5 = v4 * 0.044715
        v6 = v1 + v5
        v7 = v6 * 0.7978845608028654
        v8 = torch.tanh(v7)
        v9 = v8 + 1
        v10 = v2 * v9
        v11 = self.conv_1(v10)
        v12 = v11 * 0.5
        v13 = v11 * v11
        v14 = v13 * v11
        v15 = v14 * 0.044715
        v16 = v11 + v15
        v17 = v16 * 0.7978845608028654
        v18 = torch.tanh(v17)
        v19 = v18 + 1
        v20 = v12 * v19
        v21 = self.conv_2(v10)
        v22 = v21 * 0.5
        v23 = v21 * v21
        v24 = v23 * v21
        v25 = v24 * 0.044715
        v26 = v21 + v25
        v27 = v26 * 0.7978845608028654
        v28 = torch.tanh(v27)
        v29 = v28 + 1
        v30 = v22 * v29
        return v10, v30
# Inputs to the model
x0 = torch.randn(1, 6, 32, 32)
