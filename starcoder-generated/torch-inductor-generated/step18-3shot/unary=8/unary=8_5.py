
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(4, 6, 8, 1, 0, 0, 1, 2)
    def forward(self, x1):
        v1 = torch.sign(x1)
        v2 = self.conv_transpose(v1)
        v3 = v2 + 3
        v4 = torch.clamp(v3, min=0)
        v5 = torch.clamp(v4, max=6)
        v6 = torch.abs(v1)
        v7 = v6 * v5
        v8 = torch.abs(v7)
        v9 = v8 / 6
        return v9
# Inputs to the model
x1 = torch.randn(1, 4, 2, 2)
