
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(6, 16, 1, stride=1, padding=1)
        self.conv3 = torch.nn.Conv2d(6, 16, 1, stride=1, padding=1)
    def forward(self, x1, other=None, x4=None):
        v1 = self.conv(x1)
        v2 = self.conv3(x1)
        if other == None:
            other = torch.randn(v1.shape)
        v3 = v1 + other
        if x4 == None:
            x4 = torch.randn(x4.shape)
        v4 = v3 + x4
        return v4
# Inputs to the model
x1 = torch.randn(1, 6, 64, 64)
