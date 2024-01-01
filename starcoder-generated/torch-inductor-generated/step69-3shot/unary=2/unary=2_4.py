
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(1, 1, 3, stride=1, padding=(1, 1))
        self.conv_transpose_1 = torch.nn.ConvTranspose3d(4, 4, 3, stride=2, padding=(1, 1, 1))
        self.conv_transpose_2 = torch.nn.ConvTranspose3d(1, 2, 3, stride=(2, 1), padding=(1, 0, 1))
    def forward(self, x1):
        v1 = self.conv_transpose(x1)
        v2 = v1 * 0.5
        v3 = v1 * v1 * v1
        v4 = v3 * 0.044715
        v5 = v1 + v4
        v6 = v5 * 0.7978845608028654
        v7 = torch.tanh(v6)
        v8 = v7 + 1
        v9 = v2 * v8
        v10 = self.conv_transpose_1(v10)
        v11 = v10 * 0.5
        v12 = v10 * v10 * v10
        v13 = v12 * 0.044715
        v14 = v10 + v13
        v15 = v14 * 0.7978845608028654
        v16 = torch.tanh(v15)
        v17 = v16 + 1
        v18 = v11 * v17
        v19 = self.conv_transpose_2(v18)
        v20 = v19 * 0.5
        v21 = v19 * v19 * v19
        v22 = v21 * 0.044715
        v23 = v19 + v22
        v24 = v23 * 0.7978845608028654
        v25 = torch.tanh(v24)
        v26 = v25 + 1
        v27 = v20 * v26
        return v27
# Inputs to the model
x1 = torch.randn(8, 1, 3, 28, 28)
