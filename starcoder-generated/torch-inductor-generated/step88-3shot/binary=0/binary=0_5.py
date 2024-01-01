
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(26, 9, 10, stride=10, padding=10)
    def forward(self, x1, other=1, padding1=None, padding2=None, padding3=None):
        v1 = self.conv(x1)
        v2 = v1 + other
        if padding1 == None:
            padding1 = torch.randn(v2.shape)
        if padding2 == None:
            padding2 = torch.randn(v2.shape)
        if padding3 == None:
            padding3 = torch.randn(v2.shape)
        v3 = v2 + padding1
        return v3
# Inputs to the model
x1 = torch.randn(1, 26, 100, 100)
