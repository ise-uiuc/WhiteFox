
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 3, 2, stride=1, padding=1)
        self.swish = torch.nn.SiLU()
        self.dropout = torch.nn.Dropout2d()
    def forward(self, x1):
        t1 = self.conv(x1)
        t2 = 3 + t1
        t3 = torch.clamp(t2, 0, 6)
        t4 = t1 * t3
        t5 = t4 / 6
        t6 = self.conv(t5)
        t7 = t2 + 3
        t8 = torch.clamp(t7, 0, 6)
        t9 = t7 * t8
        t10 = t9 / 6
        t11 = t5 + t10
        return t6
# Inputs to the model
x1 = torch.randn(2, 3, 64, 64)
