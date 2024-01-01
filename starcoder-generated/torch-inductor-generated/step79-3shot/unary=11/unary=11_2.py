
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(64, 64, 3, stride=2, padding=1)
        self.conv_transpose1 = torch.nn.ConvTranspose2d(64, 64, 3, stride=2, padding=1)
    def forward(self, x1):
        v1 = self.conv_transpose(x1)
        v2 = self.conv_transpose1(v1)
        v3 = v2 + 3
        v4 = torch.clamp_min(v3, 0)
        v5 = torch.clamp_max(v4, 6)
        v6 = v5 / 6
        return v6
# Inputs to the model
x1 = torch.randn(1, 64, 128, 128)
