
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.neg_slope = torch.nn.Parameter(-0.1)
        self.conv_t1 = torch.nn.ConvTranspose2d(48, 64, 2, stride=2, padding=0)
        self.conv_t2 = torch.nn.ConvTranspose2d(64, 64, 2, stride=2, padding=0)
        self.conv_t3 = torch.nn.ConvTranspose2d(64, 64, 2, stride=1, padding=0)
        self.conv_t4 = torch.nn.ConvTranspose2d(64, 24, 1, stride=1, padding=0)
        self.conv_t5 = torch.nn.ConvTranspose2d(24, 24, 2, stride=1, padding=0)
        self.conv_t6 = torch.nn.ConvTranspose2d(24, 19, 1, stride=1, padding=0)
        self.conv_t8 = torch.nn.ConvTranspose2d(19, 48, 2, stride=1, padding=0)
        self.conv_t9 = torch.nn.ConvTranspose2d(48, 21, 1, stride=1, padding=0)
        self.pool_0 = torch.nn.AvgPool2d(2, 2, 1)
    def forward(self, x6):
        x7 = self.conv_t1(x6)
        x8 = self.conv_t2(x7)
        x9 = self.conv_t3(x8)
        x10 = self.conv_t4(x9)
        x11 = self.conv_t5(x10)
        x12 = self.conv_t6(x11)
        x13 = self.neg_slope * x12
        x14 = x12 > 0
        x15 = x12 * self.neg_slope
        x16 = torch.where(x14, x12, x15)
        x17 = x16 * -0.9357
        x18 = self.pool_0(x13)
        x19 = torch.cat((x17, x18), 1)
        x20 = self.conv_t8(x19)
        y1 = self.conv_t9(x20)
        return torch.nn.functional.interpolate(y1, scale_factor=[4.0, 2.0])
# Inputs to the model
x6 = torch.randn(1, 48, 28, 28)
