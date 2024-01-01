
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.pointwise_conv2d = torch.nn.Conv2d(6, 6, (48, 16), stride=(12, 8))
        self.conv_transpose = torch.nn.ConvTranspose2d(6, 24, (26, 8), stride=(11, 3))
    def forward(self, x1):
        v1 = self.pointwise_conv2d(x1)
        v2 = v1 * 0.5
        v3 = v1 * v1 * v1
        v4 = v3 * 0.044715
        v5 = v1 + v4
        v6 = v5 * 0.7978845608028654
        v7 = torch.tanh(v6)
        v8 = v7 + 1
        v9 = v2 * v8
        v10 = v1 + v4
        v11 = v10 * 0.7978845608028654
        v12 = torch.tanh(v11)
        v13 = v12 + 1
        v14 = v2 * v13
        v15 = v1 + v4
        v16 = torch.tanh(v15)
        v17 = v16 + 3
        v18 = v2 * v17
        v19 = [v9, v14, v18]
        v20 = v19[1]
        v21 = v19[2]
        out1 = v20 + v21
        out2 = v14 * v18
        out3 = torch.stack(v19, dim=0)
        return out3, out1, out2
# Inputs to the model
x1 = torch.randn(2, 2, 208, 336)
