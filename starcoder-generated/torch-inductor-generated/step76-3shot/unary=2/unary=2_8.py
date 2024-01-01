
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(3, 3, kernel_size=(3, 5), stride=(3, 5))
        self.convtranspose2 = torch.nn.ConvTranspose2d(3, 2, 1, stride=1, padding=0)
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
        v10 = self.convtranspose2(v9)

        v11 = v10 * 1.0
        v12 = v11 * 0.0
        v13 = v11 * v11
        v14 = v13 * 1.0
        v15 = v12 + v14
        v16 = v15 * 1.0
        v17 = v16 + 1.0
        v18 = v11 * v17
        return v18
# Inputs to the model
x1 = torch.randn(2, 3, 8, 8)
