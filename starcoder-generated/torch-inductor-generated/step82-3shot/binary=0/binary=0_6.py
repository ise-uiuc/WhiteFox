
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(16, 16, 3, stride=1, dilation=1)
    def forward(self, x1, other=1, padding0=None, padding1=None, padding2=1, padding3=None):
        v1 = self.conv(x1)
        if padding0 == None:
            padding0 = torch.randn(v1.shape)
        v2 = v1 + other
        return v2
# Inputs to the model
x1 = torch.randn(1, 16, 80, 80)
