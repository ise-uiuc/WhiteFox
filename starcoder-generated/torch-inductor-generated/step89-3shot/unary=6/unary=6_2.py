
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 45, 2, stride=1, padding=1)
        self.b1 = torch.nn.BatchNorm2d(45)
        self.b2 = torch.nn.BatchNorm2d(45)
        self.b3 = torch.nn.BatchNorm2d(45)
        self.b4 = torch.nn.BatchNorm2d(45)
        self.b5 = torch.nn.BatchNorm2d(45)
    def forward(self, x1):
        t1 = self.conv(x1)
        t2 = self.b1(t1)
        t3 = self.b2(t1)
        t4 = self.b3(t1)
        t5 = self.b4(t1)
        t6 = self.b5(t1)
        t7 = t2 + t3 + t4 + t5 + t6
        t8 = t7 > 0
        t9 = t7 * t8 + t7 * (1 - t8)
        t10 = t9 * 2
        t11 = t10 > 0
        t12 = t10 * t11 + t10 * (1 - t11)
        t13 = t12 * 0.5
        t14 = t13 > 0
        t15 = t13 * t14 + t13 * (1 - t14)
        t16 = t15 * 1.25
        t17 = t16 > 0
        t18 = t16 * t17 + t16 * (1 - t17)
        t19 = t18 * 2
        t20 = t19 * 2
        t21 = t20 > 0
        t22 = t20 * t21 + t20 * (1 - t21)
        t23 = t22 * 0.05
        t24 = t23 * 2
        t25 = t24 * 2
        t26 = t25 > 0
        t27 = t25 * t26 + t25 * (1 - t26)
        t28 = t27 * 0.008333
        t29 = t28 * 2
        t30 = t29 * 2
        t31 = t30 * 2
        t32 = t31 * 2
        t33 = t32 * 2
        t34 = t33 * 0.0013888
        t35 = t34 * 0.000006944
        t36 = t24 * 2
        t37 = t36 * 2
        t38 = t37 > 0
        t39 = t37 * t38 + t37 * (1 - t38)
        t40 = t39 * 0.05
        t41 = t36 * 2
        t42 = t41 * 2
        t43 = t42 > 0
        t44 = t42 * t43 + t42 * (1 - t43)
        t45 = t44 * 0.05
        t46 = t41 * 2
        t47 = t46 > 0
        t48 = t46 * t47 + t46 * (1 - t47)
        t49 = t48 * 1.25
        t50 = t46 * 2
        t51 = t50 > 0
        t52 = t50 * t51 + t50 * (1 - t51)
        t53 = t52 * 0.125
        t54 = t4 * 0.03125
        t55 = t12 * 16
        t56 = t16 * 128
        t57 = t19 * 1024
        t58 = t22 * 4096
        t59 = t25 * 32768
        t60 = t28 * 49152
        t61 = t31 * 1048576
        t62 = 1 + self.conv(x1)
        t63 = t10 + t13 + t16 + t19 + t22 + t24 + t25 + t26 + t27 + t28 + t29 + t30 + t31 + t32 + t33 + t34 + t35 + t36 + t37 + t38 + t39 + t40 + t41 + t42 + t43 + t44 + t45 + t46 + t47 + t48 + t49 + t50 + t51 + t52 + t53 + t54 + t55 + t56 + t57 + t58 + t59 + t60 + t61
        return t35 + t62 + t63
x1 = torch.randn(1, 3, 224, 224)
