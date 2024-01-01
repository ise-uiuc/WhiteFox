
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.negative_slope = 4.56
        self.conv_t = torch.nn.ConvTranspose2d(out_channels=256, kernel_size=(3, 3), stride=(2, 2), padding=1, groups=1, dilation=1, output_padding=1, padding_mode='zeros')
        self.conv = torch.nn.Conv2d(in_channels=256, out_channels=32, kernel_size=(3, 3), stride=(1, 1), padding=1, dilation=1)
    def forward(self, input):
        t0 = input + 3.5
        t1 = t0 * -0.753426845
        t2 = t1.permute(0, 3, 1, 2)
        t3 = t2.contiguous()
        t4 = t3.detach()
        t5 = self.conv_t(t4)
        t6 = t5 > 0
        t7 = t5 * self.negative_slope
        t8 = torch.where(t6, t5, t7)
        t9 = t8.mean((-1, -2), True)
        t10 = t9.reshape(((1, ) + t9.size()))
        t11 = t10 * 0.47788154
        t12 = self.conv(t11)
        t13 = t12 * 2.20276678
        return t13
# Inputs to the model
input = torch.randn(3, 256, 40, 40)
