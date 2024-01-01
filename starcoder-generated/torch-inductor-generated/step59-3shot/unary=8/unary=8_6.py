
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 6, 3, stride=2, padding=1, bias=False, dilation=2)
        self.conv_transpose1 = torch.nn.ConvTranspose2d(6, 2, 2, stride=2, padding=2, dilation=1, output_padding=1) # stride 2 is not compatible with output_padding
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = self.conv_transpose1(v1)
        v3 = v2 + 3
        v4 = torch.clamp(v3, min=0)
        v5 = torch.clamp(v4, max=6)
        v6 = v2 * v5
        v7 = v6 / 6
        return v7
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
