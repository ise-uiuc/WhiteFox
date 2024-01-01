
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(12, 7, 1, stride=1, padding=1)
    def forward(self, x1, other=0.1, padding1=None, padding2=None, padding3=None, other1=0.1):
        v1 = self.conv(x1)
        if padding1 == None:
            padding1 = torch.randn(v1.shape)
        if padding2 == None:
            padding2 = torch.randn(v1.shape)
        if padding3 == None:
            padding3 = torch.randn(v1.shape)
        v2 = v1 + other
        v3 = v2 + 0.1
        v4 = v3 + 0.1
        v5 = v4 + 0.1
        v6 = v5 + 0.1
        v7 = v6 + other1
        return v7
# Inputs to the model
x1 = torch.randn(1, 12, 20, 20)
