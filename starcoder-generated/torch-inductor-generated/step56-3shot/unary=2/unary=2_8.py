
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose3d(4, 2, (9, 2, 8), stride=(8, 5, 7), padding=(2, 7, 5), output_padding=(4, 5, 1), groups=1, bias=True) # Change stride, padding, and output_padding in constructor
    def forward(self, x2):
        v1 = self.conv_transpose(x2)
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
x2 = torch.randn(2, 4, 13, 12, 12)
