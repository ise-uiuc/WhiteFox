
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv0 = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
        self.conv1 = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
    def forward(self, x1, x2=None, padding0=None, padding1=None, padding2=None):
        v1 = self.conv0(x1)
        if x2 is None:
            x2 = torch.randn(v1.shape)
        v2 = v1 + x2
        if padding0 == None:
            padding0 = torch.randn(v2.shape)
        v3 = v2 + padding0
        v4 = self.conv1(x1)
        if padding1 == None:
            padding1 = torch.randn(v4.shape)
        v5 = v3 + padding1
        if padding2 == None:
            padding2 = torch.randn(v5.shape)
        v6 = v4 + padding2
        return v6
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
