
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 3, stride=2, padding=1)
    def forward(self, x1, padding1=None, padding2=None, padding3=None, padding4=None):
        v1 = self.conv(x1)
        if padding1 == None:
            padding1 = torch.randn(v1.shape)
        if padding2 == None:
            padding2 = torch.randn(v1.shape)
        if padding3 == None:
            padding3 = torch.randn(v1.shape)
        if padding4 == None:
            padding4 = torch.randn(v1.shape)
        v2 = v1
        return v2
# Inputs to the model
x1 = torch.randn(3, 3, 10, 10)
