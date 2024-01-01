
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 7, 1, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(7, 8, 1, stride=1, padding=1)
        self.conv3 = torch.nn.Conv2d(8, 2, 1, stride=1, padding=1)
    def forward(self, x1, other=1.7, padding1=None):
        var1 = self.conv1(x1)
        if not None in (padding1, padding2):
            var1 += padding1
            var1 -= padding2
        var2 = self.conv2(var1)
        if not None in (padding1, padding2):
            var2 += padding1
        var3 = self.conv3(var2)
        v2 = var3 - other
        return v2
# Inputs to the model
x1 = torch.randn(1, 3, 32, 32)
