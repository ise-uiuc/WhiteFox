
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
    def forward(self, x1):
        t1 = 3
        t2 = self.conv(x1)
        t3 = t1 + t2
        t4 = t3.clamp_min(0)
        t5 = t4.clamp_max(6)
        t6 = t5.div(6)
        return t6
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
