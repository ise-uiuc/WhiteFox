
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t1 = torch.nn.ConvTranspose2d(94, 84, 3, stride=2, bias=False)
        self.conv_t2 = torch.nn.ConvTranspose2d(84, 144, 3, stride=2, padding=1, bias=False)
        self.conv_t3 = torch.nn.ConvTranspose2d(144, 160, 3, stride=2, bias=False)
        self.conv_t4 = torch.nn.ConvTranspose2d(160, 160, 1, stride=2, bias=False)
    def forward(self, x1):
        t1 = self.conv_t1(x1)
        t2 = t1 > 0
        t3 = t1 * -0.042
        t4 = torch.where(t2, t1, t3)
        t5 = self.conv_t2(t4)
        t6 = t5 > 0
        t7 = t5 * 0.302
        t8 = torch.where(t6, t5, t7)
        t9 = self.conv_t3(t8)
        t10 = t9 > 0
        t11 = t9 * -0.261
        t12 = torch.where(t10, t9, t11)
        t13 = self.conv_t4(t12)
        t14 = t13 > 0
        t15 = t13 * -0.083
        t16 = torch.where(t14, t13, t15)
        return t16
# Inputs to the model
x1 = torch.randn(16, 94, 12, 18)
