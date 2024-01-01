
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose1 = torch.nn.ConvTranspose2d(8, 7, 3, stride=1, padding=1, dilation=1)
        self.conv_transpose2 = torch.nn.ConvTranspose2d(7, 1, 8, stride=2, padding=2, dilation=2)
    def forward(self, x1):
        v1 = self.conv_transpose1(x1)
        v2 = v1 * 0.5
        v3 = v1 * 0.7071067811865476
        v4 = torch.erf(v3)
        v5 = v4 + 1
        v6 = v2 * v5
        v7 = self.conv_transpose2(v6)
        return v7
# Inputs to the model
x1 = torch.randn(1, 8, 64, 64)
