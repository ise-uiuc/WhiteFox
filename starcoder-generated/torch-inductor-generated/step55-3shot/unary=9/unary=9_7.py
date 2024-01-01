
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(8, 8, 9)
    def forward(self, x):
        t1 = self.conv(x)
        t2 = 3 + t1
        t3 = t2.clamp(min=0)
        t4 = t3.clamp(max=6)
        t5 = t4.div(6)
        v6 = self.conv2(t5)
        t7 = 3 + v6
        t8 = t7.clamp(min=0)
        t9 = t8.clamp(max=6)
        t10 = t9.div(6)
        return t10
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
