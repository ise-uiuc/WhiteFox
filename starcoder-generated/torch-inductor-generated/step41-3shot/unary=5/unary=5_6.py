
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(3, 1, 3, stride=2, padding=1)
        self.conv = torch.nn.Conv2d(1, 1, 3, stride=2, padding=1)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = self.conv_transpose(v1)
        v3 = v1 * 0.5
        v4 = v2 * 0.5
        v5 = v1 * 0.7071067811865476
        v6 = v2 * 0.7071067811865476
        v7 = torch.erf(v5)
        v8 = torch.erf(v6)
        v9 = v7 + 1
        v10 = v8 + 1
        v11 = v3 * v9
        v12 = v4 * v10
        v13 = v12 + v11
        return v13
# Inputs to the model
x1 = torch.randn(1, 3, 224, 224)
