
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1, bias=False)
    def forward(self, x1, other=1):
        v1 = self.conv(x1)
        v2 = v1 + other
        return v2


def padding_computation(x1, other):
    v1 = torch.nn.functional.pad(x1, [1]*6)
    v2 = v1 + other
    return v2
# Inputs to the model
x1 = torch.randn(1, 3, 32, 32)

# Model begins
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1, bias=False)
    def forward(self, x1, other=1):
        v1 = self.conv(x1)
        v2 = padding_computation(v1, other)
        return v2
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
