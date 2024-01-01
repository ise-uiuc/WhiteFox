
class Model(torch.nn.Module):
    def __init__(self, negative_slope):
        super().__init__()
        self.conv_t1 = torch.nn.ConvTranspose2d(1, 1048, 3, stride=1, padding=1, bias=False)
        self.conv_t2 = torch.nn.ConvTranspose2d(1048, 512, 1, stride=1, bias=False)
        self.conv_t3 = torch.nn.ConvTranspose2d(512, 128, 1, stride=1, bias=False)
        self.conv_t4 = torch.nn.ConvTranspose2d(128, 3, 3, stride=1, padding=1, bias=False)
        self.negative_slope = negative_slope
    def forward(self, x42):
        t1 = self.conv_t1(x42)
        t2 = t1 > 0
        t3 = t1 * self.negative_slope
        t4 = torch.where(t2, t1, t3)
        t5 = self.conv_t2(t4)
        t6 = t5 > 0
        t7 = t5 * self.negative_slope
        t8 = torch.where(t6, t5, t7)
        t9 = self.conv_t3(t8)
        t10 = t9 > 0
        t11 = t9 * self.negative_slope
        t12 = torch.where(t10, t9, t11)
        t13 = self.conv_t4(t12)
        t14 = t13 > 0
        t15 = t13 * self.negative_slope
        t16 = torch.where(t14, t13, t15)
        return torch.nn.functional.softmax(t16)
negative_slope = 0.109
# Inputs to the model
x42 = torch.randn(1, 1, 178, 675)
