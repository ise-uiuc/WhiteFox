
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(9, out_channels=17, kernel_size=11, stride=17, padding=0, dilation=1, output_padding=0, groups=1, bias=True, padding_mode='zeros')
    def forward(self, x1):
        v1 = self.conv_transpose(x1)
        v2 = torch.sigmoid(v1)
        return v2
# Inputs to the model
x1 = torch.randn(1, 9, 23, 29)
