
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(32, 32, 7, stride=15, dilation=1)
    def forward(self, x, padding0=None, padding1=None, padding2=1, padding3=None, t2_shape=1):
        v1 = self.conv(x)
        if padding0 == None:
            padding0 = torch.randn(v1.shape)
        v2 = v1 + padding0
        return v2
# Inputs to the model
x = torch.randn(1, 32, 304, 304)
