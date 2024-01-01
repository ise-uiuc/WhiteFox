
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(3, 8, kernel_size=5, padding=0, output_padding=0, bias=False)
        self.conv = torch.nn.Conv2d(3, 2, kernel_size=5, padding=2)
    def forward(self, x1, x2):
        x2 = self.conv(x2)
        v1 = self.conv_transpose(x1)
        v2 = v1 + x2
        v3 = torch.clamp(v2, min=0)
        v4 = torch.clamp(v3, max=6)
        v5 = v1 * v4
        v6 = v5 / 6
        return v6
# Inputs to the model
x1 = torch.randn(1, 3, 16, 16)
x2 = torch.randn(1, 3, 8, 8)
