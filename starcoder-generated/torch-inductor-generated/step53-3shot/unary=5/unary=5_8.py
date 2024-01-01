
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__(5, 12, 18)
        self.conv_transpose = torch.nn.ConvTranspose2d(5, 12, 18, padding=(1, 2), dilation=(3, 4))
    def forward(self, x1):
        v1 = self.conv_transpose(x1)
        v2 = v1 * 0.5
        v3 = v1 * 0.7071067811865476
        v4 = torch.erf(v3)
        v5 = v4 + 1
        v6 = v2 * v5
        return v6
# Inputs to the model
x1 = torch.randn(1, 5, 64, 64)
