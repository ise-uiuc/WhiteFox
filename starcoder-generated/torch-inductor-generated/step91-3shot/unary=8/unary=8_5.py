
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose_1 = torch.nn.ConvTranspose2d(3, 3, 3, stride=2, padding=1)
        self.conv_transpose_2 = torch.nn.ConvTranspose2d(3, 3, 3, stride=1, padding=1)
    def forward(self, x1):
        d = torch.randint(0, 1, (1,)).item()
        v1 = self.conv_transpose_1(x1) if d else self.conv_transpose_2(x1)
        v2 = v1 + 3
        v3 = torch.clamp(v2, min=0)
        v4 = torch.clamp(v3, max=6)
        v5 = v1 * v4
        v6 = v5 / 6
        return v6
# Inputs to the model
x1 = torch.randn(1, 3, 30, 30)
