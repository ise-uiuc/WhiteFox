
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 13, 2, stride=2, padding=4)
    def forward(self, x1):
        t1 = self.conv(x1)
        t2 = torch.tensor(2) + t1
        t3 = torch.clamp_min(t2, 0)
        t4 = torch.clamp_max(t3, 6)
        t5 = t1 * t4
        t6 = t5 / 6
        return t6
# Inputs to the model
x1 = torch.randn(2, 3, 28, 28)
