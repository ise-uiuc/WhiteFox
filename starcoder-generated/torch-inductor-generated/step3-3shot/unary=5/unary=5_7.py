
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(3, 8, 1, stride=1, padding=0)
        self.zeros = torch.nn.zeros(8, 38, 64, dtype=torch.float32)
        self.add = torch.add
    def forward(self, x1):
        v1 = self.conv_transpose(x1)
        v2 = v1 * 0.5
        v3 = v1 * 0.7071067811865476
        v4 = torch.erf(v3)
        v5 = v4 + 1
        v6 = self.add(v2, v5)
        v7 = v6 * v5
        return v7
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
