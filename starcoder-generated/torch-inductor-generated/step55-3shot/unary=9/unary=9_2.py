
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
        self.other_conv = torch.nn.Conv2d(8, 8, 9)
    def forward(self, x1):
        t1 = self.conv(x1)
        t2 = 3
        t3 = t2 + t1
        t4 = torch.clamp(t3, min=0)
        t5 = torch.clamp(t4, max=6)
        t6 = torch.div(t5, 6.0)
        t7 = self.other_conv(t6)
        t8 = 3
        t9 = t8 + t7
        t10 = torch.clamp(t9, min=0)
        t11 = torch.clamp(t10, max=6)
        t12 = torch.div(t11, 6.0)
        return t12
# Inputs to the model
x1 = torch.randn(7, 3, 64, 64)
