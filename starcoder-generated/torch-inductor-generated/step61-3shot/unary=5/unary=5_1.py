
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(12, 24, 2, stride=2, padding=0, bias=True)
    def forward(self, x1):
        v1 = self.conv_transpose(x1)
        v3 = self.conv_transpose.weight * 0.7071067811865476
        v5 = torch.erf(v3)
        v6 = v5 + 0.5
        v7 = v1 * v6
        return v7
# Inputs to the model
x1 = torch.randn(1, 12, 32, 32)
