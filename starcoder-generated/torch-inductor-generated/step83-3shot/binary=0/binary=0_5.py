
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(10, 20, 1, stride=1, padding=1)
    def forward(self, x1, x2=7, x3=[], padding1=[]):
        v1 = self.conv(x1)
        if x2 == 7:
            x2 = torch.randn(v1.shape)
        v2 = v1 + x2
        if x3 == []:
            x3 = torch.randn(v2.shape)
        v3 = v2 + x3
        if padding1 == []:
            padding1 = torch.randn(v3.shape)
        v4 = v3 + padding1
        return v4
# Inputs to the model
x1 = torch.randn(1, 10, 64, 64)
