
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(64, 1, 2, stride=2, dilation=1)
    def forward(self, x1, other=1, padding1=None, padding2=None):
        v1 = self.conv(x1)
        if padding2 == None:
            padding2 = torch.randn(v1.shape)
        v2 = v1 + other
        return v2
# Inputs to the model
x1 = torch.randn(1, 64, 32, 32)
