
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.avgpool = torch.nn.AvgPool2d(3, stride=1, padding=1)
        self.conv = torch.nn.Conv2d(64, 2, 1, stride=1, padding=0)
    def forward(self, x1, other=None, padding1=None):
        v1 = self.avgpool(x1)
        if other == None:
            other = torch.randn(v1.shape)
        v2 = self.conv(v1)
        v3 = v2 + other
        return v3
# Inputs to the model
x1 = torch.randn(1, 64, 64, 64)
