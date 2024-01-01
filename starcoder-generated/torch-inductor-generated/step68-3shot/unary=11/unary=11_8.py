
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose_1 = torch.nn.ConvTranspose1d(1, 1, 1, stride=2, padding=1)
        self.conv_transpose_2 = torch.nn.ConvTranspose1d(1, 1, 1, stride=2, padding=0)
    def forward(self, x1):
        v1 = self.conv_transpose_1(x1)
        v2 = self.conv_transpose_2(v1)
        v3 = v1 + 3
        v4 = v2 + v3
        v5 = torch.clamp_min(v4, 0)
        v6 = torch.clamp_max(v5, 6)
        v7 = v6 / 6
        return v7
# Inputs to the model
x1 = torch.randn(1, 1, 6)
