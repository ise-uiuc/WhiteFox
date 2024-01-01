
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose1 = torch.nn.ConvTranspose2d(6, 7, 5, stride=4, padding=6, dilation=9)
        self.conv_transpose2 = torch.nn.ConvTranspose2d(7, 8, 2, stride=2, padding=5)
    def forward(self, x1):
        v1 = x1
        v2 = self.conv_transpose1(v1)
        v3 = v2 * 0.5
        v4 = v2 * 0.7071067811865476
        v5 = torch.erf(v4)
        v6 = v5 + 1
        v7 = v3 * v6
        v8 = self.conv_transpose2(v7)
        return v8
# Inputs to the model
x1 = torch.randn(1, 6, 64, 64)
