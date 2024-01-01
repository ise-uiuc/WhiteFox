
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose1 = torch.nn.ConvTranspose2d(20, 32, 3, stride=2, padding=1)
        self.conv_transpose2 = torch.nn.ConvTranspose2d(32, 16, 1, stride=2, padding=0)
        self.conv_transpose3 = torch.nn.ConvTranspose2d(16, 2, 1, stride=2, padding=0)
    def forward(self, x1):
        v1 = self.conv_transpose1(x1)
        v2 = v1 * 0.5
        v3 = v1 * 0.7071067811865476
        v4 = torch.erf(v3)
        v5 = v4 + 1
        v6 = v2 * v5
        v7 = self.conv_transpose2(v6)
        v8 = self.conv_transpose3(v7)
        return v8
# Inputs to the model
x1 = torch.randn(1, 20, 64, 64)
