
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(12, 20, 4, stride=3, padding=1)
        self.conv_transpose = torch.nn.ConvTranspose2d(20, 32, 3, stride=2)
        self.conv_transpose_2 = torch.nn.ConvTranspose2d(32, 32, 4, stride=2, padding=1)
        self.conv_transpose_3 = torch.nn.ConvTranspose2d(32, 32, 2, stride=2, padding=1)
        self.conv_transpose_4 = torch.nn.ConvTranspose2d(32, 3, 2, stride=1, padding=0)
    def forward(self, x0):
        v0 = self.conv(x0)
        v1 = self.conv_transpose(v0)
        v2 = self.conv_transpose_2(v1)
        v4 = self.conv_transpose_4(v2)
        v3 = self.conv_transpose_3(v4)
        v5 = v3 + 3
        v6 = torch.clamp_min(v5, 0)
        v7 = torch.clamp_max(v6, 6)
        v8 = v7 / 6
        return v8
# Inputs to the model
x0 = torch.randn(1, 12, 64, 64)
