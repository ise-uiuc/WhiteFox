
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose1d(12, 14, 2, stride=1, padding=0, output_padding=0)
        self.conv_transpose_1 = torch.nn.ConvTranspose1d(3, 12, 3, stride=1, padding=0, output_padding=0)
        self.mul1 = torch.nn.QuantizedMul(scale=1, zero_point=0)
    def forward(self, x1):
        v1 = self.conv_transpose(x1)
        v2 = self.conv_transpose_1(v1)
        v3 = v2 * 0.5
        v4 = self.mul1(v1, v2)
        v5 = v4 * 0.044715
        v6 = v3 + v5
        v7 = v6 * 0.7978845608028654
        v8 = torch.tanh(v7)
        v9 = v8 + 1
        v10 = v3 * v9
        return v10
# Inputs to the model
x1 = torch.randn(6, 12, 10)
