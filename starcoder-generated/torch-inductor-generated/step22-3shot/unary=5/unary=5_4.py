
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose1 = torch.nn.ConvTranspose2d(2, 4, 3, stride=2, padding=0)
        self.conv_transpose2 = torch.nn.ConvTranspose2d(4, 4, 2, stride=2, padding=1)
    def forward(self, x1):
        v1 = self.conv_transpose1(x1)
        v2 = self.conv_transpose2(v1)
        v3 = v1 * 0.5
        v4 = v1 * 0.7071067811865476
        v5 = torch.erf(v4)
        v6 = v3 + 1
        v7 = v2 * v6
        return v7
# Inputs to the model
x1 = torch.randn(1, 2, 8, 8)
