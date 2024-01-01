
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 6, 1, stride=1, padding=0)
    def forward(self, x1):
        t1 = self.conv(x1)
        t2 = torch.clamp(t1, min=0, max=6, out=t1)
        t3 = torch.clamp(t2, min=0, max=6)
        t3 = torch.div(t3, 6, out=t3)
        return t3
# Inputs to the model
x1 = torch.randn(5, 3, 64, 64)
