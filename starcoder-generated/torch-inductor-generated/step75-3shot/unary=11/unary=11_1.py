
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(32, 64, 7, stride=2, groups=4)
        self.conv_transpose_2 = torch.nn.ConvTranspose2d(64, 64, 3, stride=2, groups=4, padding=3)
        self.conv_transpose_3 = torch.nn.ConvTranspose2d(64, 64, 1, stride=1, groups=4)
    def forward(self, x1):
        v1 = self.conv_transpose(x1)
        v2 = self.conv_transpose_2(v1)
        v3 = self.conv_transpose_3(v2)
        v4 = v3 + 3
        v5 = torch.clamp_min(v4, 0)
        v6 = torch.clamp_max(v5, 6)
        v7 = v6 / 6
        return v6
# Inputs to the model
x1 = torch.randn(1, 32, 64, 64)
