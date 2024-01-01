
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
    def forward(self, x1):
        t1 = self.conv(x1)
        t2 = 3
        t3 = t2 + t1
        t4 = torch.clamp(t3, min=0, max=6)
        t5 = torch.div(t4, 6.0)
        return t5
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
