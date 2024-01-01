
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
        self.p1 = torch.nn.AvgPool2d(1)
    def forward(self, x1, x2=None):
        v1 = self.conv(x1)
        v11 = self.p1(v1)
        if x2 == None:
            x2 = torch.randn(v11.shape)
        v2 = v11 + x2
        return v2
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
