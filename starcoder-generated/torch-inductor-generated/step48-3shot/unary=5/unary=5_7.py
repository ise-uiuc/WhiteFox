
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(1, 13, 3)
        self.bn = torch.nn.BatchNorm2d(6)
    def forward(self, x1):
        v1 = self.conv_transpose(x1).clamp(min=0)
        v2 = self.bn(v1)
        v3 = v2 * 0.7071067811865476
        v4 = torch.erf(v3)
        v5 = v4 + 1
        v6 = v2 * v5
        return v6
# Inputs to the model
x1 = torch.randn(64, 1, 32, 32)
