
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv = torch.nn.Conv2d(3, 3, 1, stride=1, padding=1)
    def forward(self, x1):
        x2 = 3 + self.conv(x1)
        x3 = torch.clamp(x2, min=0)
        x4 = torch.clamp(x3, max=6)
        x5 = x4 / 6
        return x5
# Inputs to the model
x1 = torch.randn(1, 3, 3, 3)
