
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose1 = torch.nn.ConvTranspose2d(1, 1, kernel_size=(1, 1), stride=(1, 1))
        self.conv_transpose2 = torch.nn.ConvTranspose2d(1, 1, kernel_size=(3, 3), stride=(3, 3))
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
        v10 = v9.contiguous()
        v11 = self.conv_transpose2(v10)
        v12 = v11 * 0.5
        v13 = v11 * v11 * v11
        v14 = v13 * 0.044715
        v15 = v11 + v14
        v16 = v15 * 0.7978845608028654
        v17 = torch.tanh(v16)
        v18 = v17 + 1
        v19 = v12 * v18
        v20 = v19.contiguous()
        v21 = v20.contiguous()
        return v21
# Inputs to the model
x1 = torch.randn(1, 1, 5, 5)
