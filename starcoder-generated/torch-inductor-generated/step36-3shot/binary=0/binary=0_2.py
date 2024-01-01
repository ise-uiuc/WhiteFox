
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(64, 64, 1, stride=1, padding=1)
    def forward(self, x1, other=1, padding1=None, padding2=None, padding3=None, padding4=None):
        v1 = self.conv(x1)
        if padding1 == None:
            padding1 = torch.randn(v1.shape)
        v2 = v1 + other
        v3 = torch.cat([v2, padding1])
        v4 = torch.cat([v3, padding2])
        v5 = torch.cat([v4, padding3])
        v6 = torch.cat([v5, padding4])
        return v6
# Inputs to the model
x1 = torch.randn(3, 64, 64)
