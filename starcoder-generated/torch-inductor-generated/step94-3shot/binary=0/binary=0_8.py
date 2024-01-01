
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 2, 2, stride=3, padding=4, dilation=1)
    def forward(self, x1, x2=None, x3=None):
        v1 = self.conv(x1)
        if x2 is None:
            x2 = torch.randn(v1.shape)
        v2 = v1 + x2
        return v2
# Inputs to the model
x1 = torch.randn(1, 1, 30, 30)
