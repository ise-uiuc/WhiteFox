
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose0 = torch.nn.ConvTranspose2d(3, kernel_size=2, bias=True, dilation=1, output_padding=0, padding_mode='zeros', padding=1, stride=1, groups=1)
    def forward(self, x1):
        v1 = self.conv_transpose0(x1)
        v2 = torch.sigmoid(v1)
        return v2
# Inputs to the model
x1 = torch.randn(1, 3, 256, 256)
