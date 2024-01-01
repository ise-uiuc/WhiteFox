
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose_8792 = torch.nn.ConvTranspose2d(8792, 3, 2, stride=2, padding=0, output_padding=0, dilation=1, groups=1, bias=False, padding_mode='zeros')
    def forward(self, x1):
        v1 = self.conv_transpose_8792(x1)
        v2 = torch.sigmoid(v1)
        v3 = v1 * v2
        return v3
# Inputs to the model
x1 = torch.randn(1, 8792, 57, 57)
