
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
        self.other_conv = torch.nn.Conv2d(8, 8, 1, stride=2, padding=1)
    def forward(self, x1):
        v1 = self.conv(x1)
        v1 = v1.add_(3)
        v1 = v1.clamp(0, 6)
        v1 = v1.div(6)
        v2 = self.other_conv(v1)
        v2 = v2.add(3)
        v2 = v2.clamp(min=0, max=6)
        v2 = v2.div_(6)
        return v2
# Inputs to the model
x1 = torch.randn(8, 3, 1280, 720)
