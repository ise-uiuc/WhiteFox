
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
        self.other_conv = torch.nn.Conv2d(8, 8, 1, stride=2, padding=1)
    def forward(self, x1):
        v1 = self.conv(x1)
        v3 = v1.add_(3)
        v5 = v3.clamp_(0, 6)
        v7 = v5.div_(6)
        v9 = self.other_conv(v7)
        v11 = v9.add(3)
        v13 = v11.clamp(0, 6)
        v15 = v13 / 6
        return v15
# Inputs to the model
x1 = torch.randn(5, 3, 64, 64)
