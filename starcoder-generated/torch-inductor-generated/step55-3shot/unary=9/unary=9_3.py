
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
        self.other_conv = torch.nn.Conv2d(8, 8, 9)
    def forward(self, x1):
        t1 = self.conv(x1)
        t2 = 3 + t1
        t3 = torch.clamp(t2, min=0, max=6)
        t4 = torch.div(t3, 6)
        t5 = self.other_conv(t4)
        t6 = 3 + t5
        t7 = torch.clamp(t6, min=0, max=6)
        t8 = torch.div(t7, 6)
        t9 = t8 + t4
        return t9
# Inputs to the model
x1 = torch.randn(5, 3, 64, 64)
