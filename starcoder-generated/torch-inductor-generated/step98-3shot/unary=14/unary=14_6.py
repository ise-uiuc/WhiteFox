
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose_1 = torch.nn.ConvTranspose2d(9, 17, 15, stride=15, padding=4, dilation=8)
        self.conv_transpose_2 = torch.nn.ConvTranspose2d(17, 8, 9, stride=9, padding=5)
        self.conv_transpose_3 = torch.nn.ConvTranspose2d(8, 7, 8, stride=3, padding=1)
        self.conv_transpose_4 = torch.nn.ConvTranspose2d(7, 6, 6, stride=1, padding=5)
        self.conv_transpose_5 = torch.nn.ConvTranspose2d(6, 4, 16, stride=2, padding=5, dilation=2)
        self.conv_transpose_6 = torch.nn.ConvTranspose2d(4, 24, 12, stride=2, padding=4)
        self.conv_transpose_7 = torch.nn.ConvTranspose2d(24, 13, 9, stride=2, padding=3)
        self.conv_transpose_8 = torch.nn.ConvTranspose2d(13, 0, 7, stride=1, padding=1)
    def forward(self, x1):
        u1 = self.conv_transpose_1(x1)
        u2 = torch.sigmoid(u1)
        u3 = u1 * u2
        u4 = self.conv_transpose_2(u3)
        u5 = torch.sigmoid(u4)
        u6 = u4 * u5
        u7 = self.conv_transpose_3(u6)
        u8 = torch.sigmoid(u7)
        u9 = u7 * u8
        u10 = self.conv_transpose_4(u9)
        u11 = torch.sigmoid(u10)
        u12 = u10 * u11
        u13 = self.conv_transpose_5(u12)
        u14 = torch.sigmoid(u13)
        u15 = u13 * u14
        u16 = self.conv_transpose_6(u15)
        u17 = torch.sigmoid(u16)
        u18 = u16 * u17
        u19 = self.conv_transpose_7(u18)
        u20 = torch.sigmoid(u19)
        u21 = u19 * u20
        u22 = self.conv_transpose_8(u21)
        return u22
# Inputs to the model
x1 = torch.randn(1, 9, 10, 10)
