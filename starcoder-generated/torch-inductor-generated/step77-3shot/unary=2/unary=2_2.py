
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transposed_2d_bias = torch.nn.ConvTranspose2d(1024, 1024, 3, stride=3, padding=1, output_padding=3)
    def forward(self, x1):
        v1 = self.conv_transposed_2d_bias(x1)
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
x1 = torch.randn(7, 1024, 612, 54)
