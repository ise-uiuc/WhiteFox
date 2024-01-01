
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.functional.conv2d_transpose
    def forward(self, x1):
        v1 = self.conv_transpose(x1, in_channels=4, out_channels=16, kernel_size=1, stride=1, padding=3, output_padding=2, groups=1)
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
x1 = torch.randn(1, 4, 64, 64)
