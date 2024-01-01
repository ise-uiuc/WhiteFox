
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
    def forward(self, t1):
        t2 = self.conv(t1)
        t3 = t2.clamp(min=0, max=6) / 6
        return t2 + t3
# Inputs to the model
x1 = torch.randn(1, 3, 128, 128)
