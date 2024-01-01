
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(4, 4, 3, stride=1, padding=1)
    def forward(self, x1):
        y = self.conv(x1)
        out = torch.nn.functional.interpolate(y, size=(14, 14), mode='bicubic', align_corners=False)
        return out
# Inputs to the model
x1 = torch.randn(1, 4, 27, 27)
