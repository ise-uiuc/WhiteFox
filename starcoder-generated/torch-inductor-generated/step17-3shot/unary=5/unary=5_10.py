
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose2 = torch.nn.ConvTranspose2d(4, 48, 5, stride=1, padding=2)
        self.conv_transpose1 = torch.nn.ConvTranspose2d(48, 2, 4, stride=4, padding=9)
        self.reshape0 = torch.reshape(list([5, 2, 2, 2, 3]), [5, 2, 2, 2, 3])
    def forward(self, x1):
        v1 = self.conv_transpose2(x1)
        v2 = v1 * 0.5
        v3 = v1 * 0.7071067811865476
        v4 = torch.erf(v3)
        v5 = v4 + 1
        v6 = v2 * v5
        v7 = self.conv_transpose1(v6)
        v8 = v7 * 0.5
        v9 = v7 * 0.7071067811865476
        v10 = torch.erf(v9)
        v11 = v10 + 1
        v12 = v8 * v11
        v13 = self.reshape0(v12)
        return v13
# Inputs to the model
x1 = torch.randn(1, 4, 64, 64)
