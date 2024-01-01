
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(4, 4, 3, stride=1, padding=1, bias=False, dilation=1, output_padding=4)
    def forward(self, x1):
        v1 = self.conv_transpose(x1)
        v3 = v1 + 3
        v4 = torch.clamp(v3, min=0)
        v5 = torch.clamp(v4, max=6)
        v6 = v1 * v5
        v7 = v6 / 6
        return v7
# Inputs to the model
x1 = torch.randn(1, 4, 24, 24)
