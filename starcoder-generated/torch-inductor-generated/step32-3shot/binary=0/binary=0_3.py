
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(6, 6, 1, stride=1, padding=1)
    def forward(self, x1, other=1, padding1=None, padding2=None, padding3=None):
        v1 = self.conv(x1)
        if padding3 == None:
            padding3 = torch.randn(v1.shape)
        v2 = v1 + other
        return v2
# Inputs to the model
x1 = torch.randn(1, 6, 64, 64)
