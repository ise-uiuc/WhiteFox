
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(16, 8, 7, stride=1, padding=1, groups=4)
    def forward(self, x1):
        v2 = torch.div(torch.clamp(torch.add(self.conv(x1)[0][0], 3), min=0, max=6), 6)
        return v2
# Inputs to the model
x1 = torch.randn(1, 16, 64, 64)
