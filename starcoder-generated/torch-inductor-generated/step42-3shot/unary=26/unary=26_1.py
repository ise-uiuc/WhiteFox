
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(63, 1, 3, stride=1, padding=1, bias=False, dilation=1, groups=1, output_padding=0)
    def forward(self, x2):
        v1 = self.conv_transpose(x2)
        v2 = v1 > 0
        v3 = v1 * 0.148
        v4 = torch.where(v2, v1, v3)
        return v4
# Inputs to the model
x2 = torch.randn(10, 63, 17, 16)
