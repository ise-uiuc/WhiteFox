
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose_2 = torch.nn.ConvTranspose2d(3, 64, 4, stride=2, padding=1, output_padding=0)
        self.conv_transpose_1 = torch.nn.ConvTranspose2d(64, 2, 3, stride=2, padding=1)
    def forward(self, x1):
        v1 = self.conv_transpose_2(x1)
        v2 = self.conv_transpose_1(v1)
        v3 = v2 + 3
        v4 = torch.clamp(v3, min=0)
        v5 = torch.clamp(v4, max=6)
        v6 = v2 * v5
        v7 = v6 / 6
        return v7
# Inputs to the model
x1 = torch.randn(1, 3, 21, 21)
