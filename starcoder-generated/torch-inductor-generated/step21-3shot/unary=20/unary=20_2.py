
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(in_channels=1, out_channels=4, kernel_size=3, stride=(4, 2), padding=(0, 2), dilation=(1, 5), output_padding=(2, 2), groups=1, bias=True, padding_mode='zeros')
    def forward(self, x1):
        v1 = self.conv_t(x1)
        return v1
# Inputs to the model
x1 = torch.randn(1, 1, 12, 14)
