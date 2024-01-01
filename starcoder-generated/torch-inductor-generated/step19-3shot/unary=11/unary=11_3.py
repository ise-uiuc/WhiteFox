
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose_1 = torch.nn.ConvTranspose1d(1, 9, 3, stride=2, output_padding=(1, 1))
        self.conv_transpose_2 = torch.nn.ConvTranspose1d(9, 1, 3, stride=2, padding=1)
    def forward(self, x1):
        v1 = self.conv_transpose_1(x1)
        v2 = self.conv_transpose_2(v1)
        v3 = v2 + 1
        v4 = torch.clamp_min(v3, 0)
        v5 = torch.clamp_max(v4, 6)
        v5 = torch.div(v5, 6)
        return v5
# Inputs to the model
x1 = torch.randn(1, 1, 64)
