
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(32, 64, 7, stride=1, padding=2)
    def forward(self, x1):
        v1 = self.conv_transpose(x1)
        v2 = v1 + 3
        v3 = torch.clamp_min(v2, 0)
        v4 = torch.clamp_max(v3, 0.5)
        v5 = v4 / 6
        return 2.0 - v5
# Inputs to the model
x1 = torch.randn(1, 32, 64, 64)
