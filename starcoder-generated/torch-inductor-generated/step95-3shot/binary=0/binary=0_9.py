
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(20, 20, 2, stride=2, dilation=2)
    def forward(self, x1, other, padding0, padding1=None):
        if padding1 == None:
            padding1 = torch.randn(x1.shape)
        v1 = self.conv(x1)
        v2 = v1 + other
        return v2
# Inputs to the model
x1 = torch.randn(1, 20, 80, 80)
other = 1
