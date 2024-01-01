
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose1 = torch.nn.ConvTranspose2d(32, 32, 4, stride=2, output_padding=1)
        self.conv_transpose2 = torch.nn.ConvTranspose2d(16, 16, 4, stride=2, output_padding=1)
    def forward(self, x1):
        v1 = self.conv_transpose1(x1)
        v2 = v1 + 3
        v3 = torch.clamp(v2, min=0)
        v4 = torch.clamp(v3, max=6)
        v5 = v1 * v4
        v6 = v5 / 6
        v7 = self.conv_transpose2(v6)
        v8 = v7 + 3
        v9 = torch.clamp(v8, min=0)
        v10 = torch.clamp(v9, max=6)
        v11 = v7 * v10
        v12 = v11 / 6
        return v12
# Inputs to the model
x1 = torch.randn(1, 32, 16, 16)
