
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose1d(4, 64, 2, stride=1)
        self.conv = torch.nn.ConvTranspose1d(64, 64, 2, stride=2, output_padding=1)
    def forward(self, x1):
        v1 = self.conv_transpose(x1)
        v2 = torch.tanh(v1)
        v3 = self.conv(v2)
        return v3
# Inputs to the model
x1 = torch.randn(1, 4, 19)
