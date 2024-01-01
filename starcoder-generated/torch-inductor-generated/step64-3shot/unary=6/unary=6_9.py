
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(4, 8, (7, 7), stride=(2, 2), padding=(3, 3))
        self.conv2 = torch.nn.Conv2d(8, 8, 2, stride=1, padding=1)
    def forward(self, x1):
        t1 = self.conv(x1)
        t2 = t1.add(3)
        t3 = t2.clamp(0, 6)
        t4 = t1.mul(t3)
        t5 = t4.div(6)
        t6 = torch.mean(t4, dim=1)
        t7 = t5 + t6
        t8 = self.conv2(t7)
        t9 = torch.mean(t8, dim=1)
        return t5
# Inputs to the model
x1 = torch.randn(8, 4, 177, 177)
