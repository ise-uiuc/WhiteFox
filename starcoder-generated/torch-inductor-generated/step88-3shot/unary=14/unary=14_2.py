
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose_2 = torch.nn.ConvTranspose1d(33, 35, 3, stride=1, padding=1, output_padding=1)
    def forward(self, x1):
        v1 = self.conv_transpose_2(x1)
        v2 = torch.tanh(v1)
        v3 = v1 * v2
        return v3
# Inputs to the model
x1 = torch.randn(2, 33, 98)
