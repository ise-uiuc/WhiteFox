
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(in_channels=1, out_channels=1, kernel_size=5, stride=1, groups=1, bias=True, dilation=1, padding=2, padding_mode='zeros', output_padding=0, padding_list=[], output_padding_list=[])
        self.sigmoid = torch.nn.functional.sigmoid
    def forward(self, x1):
        v1 = self.conv_t(x1)
        v2 = self.sigmoid(v1)
        return v2
# Inputs to the model
x1 = torch.randn(1, 1, 222, 5, 5)
