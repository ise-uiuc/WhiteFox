
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
        self.other_conv = torch.nn.Conv2d(8, 8, 9)
    def forward(self, x1):
        t1 = self.conv(x1)
        t2 = self.other_conv(t1)
        t3 = 3
        t4 = 3 + t1
        t5 = t3 + t4
        t6 = torch.clamp(t5, min=0, max=6)
        t7 = torch.div(t6, 6.0)
        t8 = t3 + t7
        t9 = torch.clamp(t8, min=0, max=6)
        t10 = torch.div(t9, 6.0)
        return t10
# Inputs to the model
x1 = torch.randn(5, 3, 64, 64)
