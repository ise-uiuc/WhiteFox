
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 3, 1, stride=1, padding=1)
    def forward(self, x1):
        t1 = self.conv(x1)
        t2 = t1 + 3
        t3 = torch.clamp(t2, 0, 6)
        t4 = torch.clamp(t3, 0, 6)
        t5 = t1 * t4
        return t5
# Inputs to the model
x1 = torch.randn(1, 3, 256, 256)
