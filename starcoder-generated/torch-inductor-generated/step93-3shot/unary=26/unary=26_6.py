
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(1, 4, kernel_size=3, stride=(1, 1), padding=1, dilation=2, groups=7, bias=False)
    def forward(self, x):
        aa = self.conv_t(x)
        ba = aa > 0
        aa = aa * 0
        aa = torch.where(ba, aa, aa)
        return aa
# Inputs to the model
x = torch.randn(1, 1, 2, 3)
