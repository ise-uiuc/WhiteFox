
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t1 = torch.nn.ConvTranspose2d(30, 20, 3, stride=1, padding=1, output_padding=1)
        self.conv_t2 = torch.nn.ConvTranspose2d(10, 20, 2, stride=2)
        self.conv_t3 = torch.nn.ConvTranspose2d(20, 10, 3, stride=2, padding=1, output_padding=1)
        self.conv_t4 = torch.nn.ConvTranspose2d(10, 5, 3, stride=2, padding=1, output_padding=1)
    def forward(self, x11):
        t1 = self.conv_t1(x11)
        t2 = t1 > 0
        t3 = t1 * 1
        t4 = torch.where(t2, t1, t3)
        t5 = self.conv_t2(t4)
        t6 = t5 > 0
        t7 = t5 * 1
        t8 = torch.where(t6, t5, t7)
        t9 = self.conv_t3(t8)
        t10 = t9 > 0
        t11 = t9 * 1
        t12 = torch.where(t10, t9, t11)
        t13 = self.conv_t4(t12)
        t14 = t13 > 0
        t15 = t13 * 1
        t16 = torch.where(t14, t13, t15)
        return t16
# Inputs to the model
x11 = torch.randn(6, 30, 224, 224)
