
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(7, 1, 1, stride=1, padding=1)
    def forward(self, x3):
        v1 = self.conv(x3)
        v2 = v1 - 1.3995e+24
        return v2
# Inputs to the model
x3 = torch.randn(1, 7, 15, 15)
