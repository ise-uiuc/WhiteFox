
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
    def forward(self, x1):
        t1 = self.conv(x1)
        t2 = t1 + 3
        t3 = t2
        t4 = t1 - t2
        t5 = t3 - t4
        t6 = t5 - t1
        t7 = t6 - t2
        t8 = t7 - t3
        return t8
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
