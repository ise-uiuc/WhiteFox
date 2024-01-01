
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 16, 1, stride=1, padding=1)
        self.conv1 = torch.nn.Conv2d(3, 16, 1, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(3, 16, 1, stride=1, padding=1)
    def forward(self, x1, x0=None, x2=None):
        v1 = self.conv(x1)
        if x0 == None:
            x0 = torch.randn(v1.size())
        v2 = self.conv1(x0)
        if x2 == None:
            x2 = torch.randn(v2.size())
        v3 = self.conv2(x2)
        v4 = v3 + v1
        return v4
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
