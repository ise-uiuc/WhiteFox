
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1, dilation=3, groups=1)
    def forward(self, x1):
        t1 = self.conv(x1)
        t2 = t1 - 3
        t3 = torch.nn.functional.relu(t2, inplace=True)
        t4 = torch.clamp_min(t3, 0)
        t5 = torch.clamp_max(t4, 6)
        t6 = t1 * t5
        t7 = t6 / 6
        return t7
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
