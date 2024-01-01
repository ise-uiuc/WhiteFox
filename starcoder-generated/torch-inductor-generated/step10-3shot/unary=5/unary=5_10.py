
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(1, 1, 30, stride=1, padding=15)
    def forward(self, x):
        v = self.conv_transpose(x)
        v1 = v * 0.5
        v2 = v * 0.7071067811865476
        v3 = torch.erf(v2)
        v4 = v3 + 1
        v5 = v1 * v4
        return v5
# Inputs to the model
x = torch.randn(1, 1, 128, 128)
