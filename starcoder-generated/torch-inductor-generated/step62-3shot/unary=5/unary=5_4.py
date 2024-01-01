
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(3, 2, 1, stride=1, padding=1)
        self.conv = torch.nn.Conv2d(4, 1, 1, stride=1, padding=1)
    def forward(self, x1, x2):
        v1 = self.conv_transpose(x2)
        v2 = v1 * 0.5
        v3 = v1 * 0.7071067811865476
        v4 = torch.erf(v3)
        v5 = v4 + 1
        v6 = v2 * v5
        v7 = self.conv(torch.cat([x1, v6], dim=1))
        return v7
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
x2 = torch.randn(1, 4, 64, 64)
