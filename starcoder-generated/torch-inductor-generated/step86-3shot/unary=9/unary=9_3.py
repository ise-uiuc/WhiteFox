
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_1 = torch.nn.Conv2d(24, 14, 7, stride=2, padding=3, bias=True)
        self.conv_2 = torch.nn.Conv2d(14, 24, 7, stride=1, padding=3, bias=True)
    def forward(self, x1):
        t1 = self.conv_1(x1)
        t2 = t1 + 3
        t3 = t2.clamp_min(0)
        t4 = t3.clamp_max(6)
        t5 = t4.div(6)
        t6 = self.conv_2(t5)
        t7 = t6 + 3
        t8 = t7.clamp_min(0)
        t9 = t8.clamp_max(6)
        t10 = t9.div(6)
        return t10
# Inputs to the model
x1 = torch.randn(2, 24, 56, 56)
