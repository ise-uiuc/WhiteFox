
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose1 = torch.nn.ConvTranspose2d(2, 9, 3, stride=1, padding=2)
        self.conv_transpose2 = torch.nn.ConvTranspose1d(9, 6, 4, stride=2, padding=1)
        self.conv_transpose3 = torch.nn.ConvTranspose1d(6, 2, 2, stride=1, padding=0)
    def forward(self, x1):
        t1 = self.conv_transpose1(x1)
        t2 = t1 * 0.5
        t3 = t1 * 0.7071067811865476
        t4 = torch.erf(t3)
        t5 = t4 + 1
        t6 = t2 * t5
        t7 = self.conv_transpose2(t6)
        t8 = t7 * 0.5
        t9 = t7 * 0.7071067811865476
        t10 = torch.erf(t9)
        t11 = t10 + 1
        t12 = t8 * t11
        t13 = self.conv_transpose3(t12)
        t14 = t13 * 0.5
        t15 = t13 * 0.7071067811865476
        t16 = torch.erf(t15)
        t17 = t16 + 1
        x2 = t14 * t17
        return x2
# Inputs to the model
x1 = torch.randn(1, 2, 16, 16)
