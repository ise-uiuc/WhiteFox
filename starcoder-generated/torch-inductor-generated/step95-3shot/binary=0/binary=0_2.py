
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(62, 16, 1, stride=1, padding=1, dilation=1)
    def forward(self, x1, other=1, padding0=1, padding1=1, padding2=1, padding3=None):
        v1 = self.conv(x1)
        v2 = v1 + other
        return v2
# Inputs to the model
x1 = torch.randn(1, 62, 144, 57)
