
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t1 = torch.nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1)
        self.conv_t2 = torch.nn.ConvTranspose3d(32, 8, 5, stride=(1,1,2), padding=(4,0,2), output_padding=(2,0,1), dilation=1, groups=2, bias=True)
    def forward(self, x35):
        x1 = self.conv_t1(x35)
        x2 = x1 > 0
        x3 = x1 * 0.02200345
        x4 = torch.where(x2, x1, x3)
        x5 = self.conv_t2(x4)
        x6 = x5 > 0
        x7 = x5 * -1.19420467959
        x8 = torch.where(x6, x5, x7)
        return x8
# Inputs to the model
x35 = torch.randn(1, 32, 7, 3, 9)
