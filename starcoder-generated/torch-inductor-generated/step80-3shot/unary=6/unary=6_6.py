
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 3, 3, stride=1,padding=1)
    def forward(self, x1):
        t2 = self.conv(x1)
        t1 =3 + t2
        t3 = torch.clamp(t1, 0, 6)
        t4 = t3 * 0.1
        t5 = t4 - 0.6
        t6 = t5 + t5
        t7 = t6 * t6
        t8 = t7 + t7
        t9 = t8 / 0.2
        t10 = t4 + t9
        return t10
# Inputs to the model
x1 = torch.randn(3, 3, 32, 32)
