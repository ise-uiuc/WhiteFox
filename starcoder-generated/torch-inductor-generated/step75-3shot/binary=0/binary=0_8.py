
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 1, stride=1, padding=0, groups=1)
    def forward(self, x1, other=1, bias=True):
        if bias == False:
            v1 = self.conv(x1)
        else:
            v1 = self.conv(x1, bias=other)
        result1 = v1 + other
        return result1
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
