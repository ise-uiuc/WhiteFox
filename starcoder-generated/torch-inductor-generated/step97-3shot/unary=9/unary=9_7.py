
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
    def forward(self, x1):
        t1 = self.conv(x1)
        v1 = t1 * 6 - 3
        t2 = torch.clamp(v1, min=0, max=6)
        v2 = torch.div(t2, 6)
        return v2
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
