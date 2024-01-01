
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 32, 1, stride=2, padding=0)
        self.conv_1 = torch.nn.Conv2d(32, 32, 1, stride=2, padding=0)
    def forward(self, x1):
        t1 = self.conv(x1)
        t2 = self.conv(x1)
        t3 = t2 + 3
        t4 = t1 + t3
        t5 = torch.clamp_min(t4, 0)
        t6 = t3 * t5
        return t6
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
