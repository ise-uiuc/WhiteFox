
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
    def forward(self, x1, other=1, padding1=None):
        var1 = self.conv1(x1)
        v2 = var1 + other
        if padding1 == None:
            v2 += self.conv2(v2)
        elif not None in (padding1, padding2):
            v2 -= padding1 + self.conv2(v2)
        return [v2, padding1]
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
