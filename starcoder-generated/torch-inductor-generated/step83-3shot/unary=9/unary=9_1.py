
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 4, 1, stride=1, padding=1)
    def forward(self, x1):
        t1 = self.conv(x1)
        t2 = 3 + t1
        t3 = torch.clamp(t2, 0, 6)
        t4 = t3 + 1
        return t4
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
