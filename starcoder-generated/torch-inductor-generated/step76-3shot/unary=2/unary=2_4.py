
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(14, 28, kernel_size=(1, 1), stride=(3, 3))
        self.convtranspose2 = torch.nn.ConvTranspose2d(28, 7, kernel_size=(3, 3), stride=(2, 2))
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
        v11 = v10 + 1.3322039914702371e-05
        v12 = v11 * 18544.0
        return v12
# Inputs to the model
x1 = torch.randn(7, 14, 24, 48)
