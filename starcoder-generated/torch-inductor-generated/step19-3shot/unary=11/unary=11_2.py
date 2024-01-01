
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(9, 32, 2, stride=2, padding=1)
    def forward(self, x1, x2, x3, x4):
        v0 = x0 + x1
        v1 = torch.cat((x2, x3, x4), 1)
        v2 = self.conv_transpose(v1)
        v3 = v2 + 3
        v4 = torch.clamp_min(v3, 0)
        v5 = torch.clamp_max(v4, 6)
        v6 = v5 / 6
        return v6
# Inputs to the model
x0 = torch.randn(1, 3, 28, 28)
x1 = torch.randn(1, 3, 28, 28)
x2 = torch.randn(1, 3, 28, 28)
x3 = torch.randn(1, 3, 28, 28)
x4 = torch.randn(1, 3, 28, 28)
