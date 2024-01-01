
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(75, 1, 3)
        self.conv0_relu = torch.nn.ReLU()
        self.conv0_t_1 = torch.nn.Conv2d(30, 75, 1)
        self.conv0_t_2 = torch.nn.Conv2d(75, 75, 3)
        self.conv0_t_3 = torch.nn.Conv2d(75, 75, 1)
        self.conv0_t_4 = torch.nn.Conv2d(75, 75, 3)
        self.conv0_t_5 = torch.nn.Conv2d(75, 75, 1)
        self.conv0_t_6 = torch.nn.Conv2d(75, 75, 3)
        self.conv0_t_7 = torch.nn.Conv2d(75, 75, 1)
        self.conv0_t_8 = torch.nn.Conv2d(75, 75, 3)
        self.conv_bn = torch.nn.BatchNorm2d(75, alpha=0.01, momentum=0.9, eps=1e-05)
        self.conv_relu = torch.nn.ReLU()
    def forward(self, x1):
        t1 = self.conv(x1)
        t1 = self.conv0_relu(t1)
        t2 = self.conv0_t_1(t1)
        t3 = self.conv0_t_2(t2)
        t4 = t3 + 0.1
        t4 = torch.sigmoid(t4)
        t5 = t1 * t4
        t6 = self.conv0_t_3(t5)
        t7 = self.conv0_t_4(t6)
        t8 = t7 + 0.1
        t8 = torch.sigmoid(t8)
        t9 = t5 * t8
        t10 = self.conv0_t_5(t9)
        t11 = self.conv0_t_6(t10)
        t12 = t11 + 0.1
        t12 = torch.sigmoid(t12)
        t13 = t9 * t12
        t14 = self.conv0_t_7(t13)
        t15 = self.conv0_t_8(t14)
        t16 = t15 + 0.1
        t16 = torch.sigmoid(t16)
        t17 = t13 * t16
        t18 = self.conv_bn(x1)
        t19 = t18 * t17
        return t19
# Inputs to the model
x1 = torch.randn(1, 75, 20, 25)
