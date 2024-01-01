
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.conv1 = torch.nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv2 = torch.nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv3 = torch.nn.Conv2d(32, 32, 3, stride=1, padding=1)
        self.conv4 = torch.nn.Conv2d(32, 32, 3, stride=1, padding=1)
        self.conv5 = torch.nn.Conv2d(32, 32, 1, stride=1)
        self.conv6 = torch.nn.Conv2d(32, 32, 1)
        self.conv7 = torch.nn.Conv2d(32, 32, 1)
        self.conv8 = torch.nn.Conv2d(32, 32, 3, bias=False)
        self.conv9 = torch.nn.Conv2d(32, 32, 3, dilation=2, padding=2)
        self.conv10 = torch.nn.Conv2d(32, 32, 3, stride=2, dilation=1, padding=1)
    def forward(self, x1):
        t1 = self.conv(x1)
        t2 = self.conv1(t1)
        t3 = self.conv2(t2)
        t4 = self.conv3(t3)
        t5 = self.conv4(t4)
        t6 = self.conv5(t5)
        t7 = self.conv6(t6)
        t8 = self.conv7(t7)
        t9 = t8 * t1
        t10 = t8 * t2
        t11 = t8 * t3
        t12 = self.conv8(t8)
        t13 = self.conv9(t8)
        t14 = self.conv10(t8)
        t15 = t1 + t12 + t13 + t14
        t16 = t2 + t12 + t13 + t14
        return t15, t16
# Inputs to the model
x1 = torch.randn(1, 3, 224, 224)
