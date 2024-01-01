
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.convtranspose = torch.nn.ConvTranspose1d(in_channels=in_channels, out_channels=out_channels, kernel_size=2, stride=2,
                                                     padding=0, output_padding=0, groups=1, bias=True, dilation=1)
    def forward(self, x1):
        v1 = self.convtranspose(x1)
        return v1
# Inputs to the model
x1 = torch.randn(1, 8, 10)
