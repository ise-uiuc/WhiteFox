
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 3, 3, stride=1, padding=1, dilation=1)
    def forward(self, x1, padding1=0, other=2, strides1=None):
        v1 = self.conv(x1)
        if strides1 == None:
            strides1 = torch.randn(v1.shape)
        v2 = v1 + other
        if padding1 == 0:
            padding1 = torch.randn(v2.shape)
        v3 = v2 + padding1
        return v3
# Inputs to the model
x1 = torch.randn(1, 1, 64, 64)
