
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
    def forward(self, x1):
        t1 = self.conv(x1)
        t3 = t1 + 3
        t4 = torch.clamp_max(t3, 6)
        t5 = torch.clamp_min(t4, 0)
        t6 = torch.div(t5, 6)
        return t6
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
