
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 3, stride=3, padding=5)
    def forward(self, x):
        t1 = self.conv(x)
        x = x + t1
        t2 = self.conv(x)
        x = x + t2 + t1
        t3 = self.conv(x)
        x = x + t3
        t4 = self.conv(x)
        x = x - t4
        t5 = self.conv(x)
        x = x - t5
        t6 = self.conv(x)
        x = x - t6
        t7 = self.conv(x)
        x = x - t7
        t8 = self.conv(x)
        x = x + t8
        t9 = self.conv(x)
        x = x + t9
        return x
# Inputs to the model
x = torch.randn(1, 3, 32, 32)
