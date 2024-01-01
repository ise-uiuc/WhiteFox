
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(30, 10, 9, stride=9, padding=0)
    def forward(self, x0):
        v0 = self.conv_transpose(x0)
        v1 = v0 * 0.5
        v2 = v0 * 0.7071067811865476
        v3 = torch.erf(v2)
        v4 = v3 + 1
        v5 = v1 * v4
        return v5
# Inputs to the model
x0 = torch.randn(1, 30, 32, 32)
