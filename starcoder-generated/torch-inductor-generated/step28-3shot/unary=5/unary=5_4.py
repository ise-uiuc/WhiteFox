
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(1, 1, 1, bias=True, stride=1, padding=0)
    def forward(self, x1):
        v1 = v7 = self.conv_transpose(x1)
        v8 = v7 * 0.125
        v2 = v1 * 0.5
        v3 = v1 * 0.7071067811865476
        v4 = torch.erf(v3)
        v5 = v4 + 1
        v6 = v2 * v5
        return v6
# Inputs to the model
x1 = torch.randn(1, 1, 64, 64)
