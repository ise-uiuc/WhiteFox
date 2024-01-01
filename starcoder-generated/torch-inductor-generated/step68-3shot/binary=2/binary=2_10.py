
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
    def forward(self, x):
        v = self.conv(x)
        if True:
            return v - 21
        elif False:
            return v
        elif False:
            return v
        elif False:
            return v + 124
# Inputs to the model
x = torch.randn(1, 3, 64, 32)
