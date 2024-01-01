
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(2, 5, 1, stride=1, padding=0, dilation=(2, 2), output_padding=(1, 1))
    def forward(self, x):
        v1 = self.conv_transpose(x)
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
x = torch.randn(2, 2, 10, 10)
