
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 3, 1, stride=1, padding=1)
    def forward(self, x1):
        t1 = self.conv(x1)
        t2 = t1 + (6.5)
        t3 = torch.clamp(t2, min=3.4, max=11.9)
        t4 = t1 * t3
        t5 = t4 / (15.5)
        return t5
# Inputs to the model
x1 = torch.randn(1, 3, 128, 128)
