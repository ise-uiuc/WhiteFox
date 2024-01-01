
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(3, 9, 1, bias=False)
        self.mul = torch.nn.functional.gelu
        self.mul_1 = torch.nn.functional.gelu
    def forward(self, x1):
        v1 = self.conv_transpose(x1)
        v2 = self.mul(v1)
        v3 = v2 * 0.5
        v4 = v2 * 0.7071067811865476
        v5 = torch.erf(v4)
        v6 = v5 + 1
        v7 = v3 * v6
        return v7
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
