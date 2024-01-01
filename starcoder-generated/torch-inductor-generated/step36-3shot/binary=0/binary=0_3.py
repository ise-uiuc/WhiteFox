
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 19, 1, stride=1, padding=1)
    def forward(self, x1, other=1, padding1=None, padding2=None, padding3=None, other2=None):
        v1 = self.conv(x1)
        if padding1 == None:
            padding1 = torch.randn(v1.shape)
        if padding2 == None:
            padding2 = torch.randn(v1.shape)
        if padding3 == None:
            padding3 = torch.randn(v1.shape)
        if other2 == None:
            other2 = torch.randn(v1.shape)
        v2 = v1 + other
        v3 = torch.cat([padding2, v2])
        v4 = torch.cat([padding1, v3])
        v5 = torch.cat([v3, v4])
        v6 = torch.cat([other2, v3])
        return v6, v5
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
