
class TransposeConv2d(torch.nn.ConvTranspose2d):
    def __init__(self):
        super().__init__(8, 6, 2)
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.trans_conv2d = TransposeConv2d()
    def forward(self, x1):
        v1 = self.trans_conv2d(x1)
        v2 = v1 * 0.5
        v3 = v1 * v1 * v1
        v4 = v3 * 0.044715
        v5 = v1 + v4
        v6 = v5 * 0.7978845608028654
        v7 = torch.tanh(v6)
        v8 = v7 + 1
        v9 = v2 * v8
        return v9
# Inputs to the model
x1 = torch.randn(1, 6, 1, 1)
