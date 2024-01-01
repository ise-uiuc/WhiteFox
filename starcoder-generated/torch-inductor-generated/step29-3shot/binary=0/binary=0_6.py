
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 7, 2, stride=2, padding=2)
    def forward(self, x1, other=1, padding1=None, padding2=None, padding3=None, padding4=None, padding5=None):
        v1 = self.conv(x1)
        if padding1 == None:
            padding1 = torch.randn(v1.shape)
        if padding2 == None:
            padding2 = torch.randn(v1.shape)
        if padding3 == None:
            padding3 = torch.randn(v1.shape)
        if padding4 == None:
            padding4 = torch.randn(v1.shape)
        if padding5 == None:
            padding5 = torch.randn(v1.shape)
        v2 = v1 + other
        return v2
# Inputs to the model
x1 = torch.randn(1, 1, 7, 7)
