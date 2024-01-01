
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(35, 52, 1, dilation=8)
    def forward(self, x1, other=None, padding1=None, padding2=1, dilation1=None):
        v1 = self.conv(x1)
        v2 = v1 + other
        return v2
# Inputs to the model
x1 = torch.randn(1, 35, 9, 9)
