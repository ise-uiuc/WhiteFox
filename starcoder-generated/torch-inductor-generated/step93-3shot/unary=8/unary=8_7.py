
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(64, 4, 3, stride=1, padding=0, output_padding=0, dilation=1)
    def forward(self, x1):
        v1 = self.conv_transpose(x1)
        v1 = v1 + 3
        v2 = torch.clamp(v1, min=0)
        v3 = torch.clamp(v2, max=6)
        v4 = v3 * v3 / 6
        return v4
# Inputs to the model
x1 = torch.randn(1, 64, 2, 3)
