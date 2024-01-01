
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(13, 4, 3, stride=2, padding=1, dilation=2)
    def forward(self, x1, other=1.37, dilation1=None, groups1=None):
        v1 = self.conv(x1)
        if dilation1 == None:
            dilation1 = torch.randn(v1.shape)
        v2 = v1 + other
        return v2
# Inputs to the model
x1 = torch.randn(13, 3, 128, 128)
