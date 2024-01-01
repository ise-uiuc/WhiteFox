
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1, bias=False)
        self.add = torch.add
    def forward(self, x1, other=1, padding1=None, padding2=None, dilation1=None):
        v1 = self.conv(x1)
        v2 = self.add(other, v1)
        return v2
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
