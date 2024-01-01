
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose1 = torch.nn.ConvTranspose2d(20, 24, 2, stride=2, padding=0, output_padding=0, dilation=1, groups=1)
        self.conv_transpose2 = torch.nn.ConvTranspose2d(24, 20, 3, stride=1, padding=1, output_padding=0, dilation=1, groups=1)
    def forward(self, x1):
        v1 = self.conv_transpose1(x1)
        v2 = v1 + 3
        v3 = torch.clamp(v2, min=0)
        v4 = torch.clamp(v3, max=6)
        v5 = self.conv_transpose2(v4)
        v6 = v5 + 3
        v7 = torch.clamp(v6, min=0)
        v8 = torch.clamp(v7, max=6)
        v9 = v5 * v8
        v10 = v9 / 6
        return v10
# Inputs to the model
x1 = torch.randn(1, 20, 32, 32)
