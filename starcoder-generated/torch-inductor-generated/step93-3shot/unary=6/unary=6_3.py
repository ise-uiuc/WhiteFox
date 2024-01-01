
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 5, 1, stride=1, padding=1)
    def forward(self, x1):
        t1 = self.conv(x1)
        t2 = torch.nn.functional.relu(t1)
        t3 = self.conv(x1)
        t4 = torch.nn.functional.tanh(t3)
        t5 = t2 + t4
        t6 = torch.nn.functional.sigmoid(t5)
        t7 = self.conv(t6)
        t8 = torch.nn.functional.mish(t7)
        t9 = t2 + t8
        t10 = t9 * t9
        s1 = 1
        t11 = torch.nn.functional.hardswish(x1 + t10, s1)
        return t11
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
