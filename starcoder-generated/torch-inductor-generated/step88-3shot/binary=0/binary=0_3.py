
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1024, 256, 1, stride=1, padding=0)
    def forward(self, x2, other=None, padding1=None, padding2=None, padding3=None):
        v1 = self.conv(x2)
        v2 = v1 + v1
        if other == None:
            other = torch.randn(v2.shape)
        v3 = v2 + other
        return v3
# Inputs to the model
x2 = torch.randn(1, 1024, 14, 14)
