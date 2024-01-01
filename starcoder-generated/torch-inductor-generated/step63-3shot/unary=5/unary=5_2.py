
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose1 = torch.nn.ConvTranspose2d(7, 9, 9, stride=3, padding=1, dilation=1)
        self.conv_transpose2 = torch.nn.ConvTranspose2d(9, 5, 3, stride=2, padding=0)
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
x1 = torch.randn(1, 6, 32, 255)
