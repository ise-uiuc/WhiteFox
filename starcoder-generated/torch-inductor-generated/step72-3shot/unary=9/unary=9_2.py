
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
    def forward(self, x1):
        v1 = self.conv(x1)
        t1 = v1.clone()
        t1.add_(3)
        v2 = t1.clamp_(min=0, max=6)
        t2 = v2.clone()
        t2.div_(6)
        return t2
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
