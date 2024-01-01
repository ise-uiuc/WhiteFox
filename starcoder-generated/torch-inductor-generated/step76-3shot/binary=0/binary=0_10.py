
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(23, 54, 14, stride=15, padding=0, dilation=1)
    def forward(self, x1, other=1, padding1=34, padding2=None):
        v1 = self.conv(x1)
        if padding2 == None:
            padding2 = torch.randn(v1.shape)
        v2 = other + v1
        return v2
# Inputs to the model
x1 = torch.randn(18, 23, 65, 66)
