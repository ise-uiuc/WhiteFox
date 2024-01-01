
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(32, 2, 3, stride=1, padding=3)
        self.conv = torch.nn.Conv2d(1, 64, 3, stride=3, padding=1)
    def forward(self, x1, x2):
        v1 = self.conv_transpose(x1)
        v2 = torch.cat((v1, x2), dim=1)
        v3 = self.conv(v2)
        v4 = v3 / 128
        v5 = torch.clamp_max(v4, 6)
        v6 = v5 + 20
        v7 = torch.clamp_min(v6, 5)
        v8 = torch.round(v7)
        return v8
# Inputs to the model
x1 = torch.randn(1, 64, 32, 32)
x2 = torch.randn(1, 1, 32, 32)
