
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose_1 = torch.nn.ConvTranspose2d(8, 6, 5, stride=1, padding=2, dilation=2)
        self.conv_transpose_2 = torch.nn.ConvTranspose2d(6, 6, 3, stride=1, padding=1, dilation=1)
        self.conv_transpose_3 = torch.nn.ConvTranspose2d(6, 5, 3, stride=1, padding=1, dilation=1)
        self.conv_transpose_4 = torch.nn.ConvTranspose2d(5, 4, 3, stride=1, padding=1, dilation=1)
        self.conv_transpose_5 = torch.nn.ConvTranspose2d(4, 4, 2, stride=1, padding=0, dilation=1)
        self.conv_transpose_6 = torch.nn.ConvTranspose2d(4, 3, 2, stride=1, padding=0, dilation=1)
    def forward(self, x1):
        t1 = self.conv_transpose_1(x1)
        t2 = torch.sigmoid(t1)
        t3 = t1 * t2
        t4 = self.conv_transpose_2(t3)
        t5 = torch.sigmoid(t4)
        t6 = t4 * t5
        t7 = self.conv_transpose_3(t6)
        t8 = torch.sigmoid(t7)
        t9 = t7 * t8
        t10 = self.conv_transpose_4(t9)
        t11 = torch.sigmoid(t10)
        t12 = t10 * t11
        t13 = self.conv_transpose_5(t12)
        t14 = torch.sigmoid(t13)
        t15 = t13 * t14
        t16 = self.conv_transpose_6(t15)
        t17 = torch.sigmoid(t16)
        t18 = t16 * t17
        return t18
# Inputs to the model
x1 = torch.randn(1, 8, 6, 5)
