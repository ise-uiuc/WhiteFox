
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(46, 16, 1, stride=1, padding=0)
        self.dropout = torch.nn.Dropout2d(p=0.75)
    def forward(self, x1):
        t1 = self.conv(x1)
        t2 = self.dropout(t1)
        t3 = self.conv(t2)
        t4 = torch.cat([t2, t3], 1)
        t5 = self.conv(t4)
        t6 = torch.randn(1, 3, 224, 224)
        t7 = torch.cat([t5, t6], 1)
        t8 = self.conv(t7)
        t9 = torch.randn(1, 3, 224, 224)
        t10 = torch.cat([t8, t9], 1)
        t11 = self.conv(t10)
        t12 = torch.randn(1, 17, 224, 224)
        t13 = torch.cat([t11, t12], 1)
        t14 = self.conv(t13)
        t15 = self.conv(t14)
        t16 = self.conv(t15)
        t17 = torch.randn(1, 26, 224, 224)
        t18 = torch.cat([t16, t17], 1)
        return self.conv(t18)
# Inputs to the model
x1 = torch.randn(1, 46, 224, 224)
