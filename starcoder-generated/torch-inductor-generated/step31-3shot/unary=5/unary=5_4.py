
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose_1 = torch.nn.ConvTranspose2d(4, 64, 8, stride=8, padding=4)
        self.conv_transpose_2 = torch.nn.ConvTranspose2d(1, 32, 8, stride=1, padding=8)
    def forward(self, x1):
        t1 = self.conv_transpose_2(self.conv_transpose_1(x1))
        v1 = t1 * 0.5
        v2 = t1 * 0.7071067811865476
        v3 = torch.erf(v2)
        v4 = v3 + 1
        v5 = v1 * v4
        return v5
# Inputs to the model
x1 = torch.randn(1, 1, 255, 80)
