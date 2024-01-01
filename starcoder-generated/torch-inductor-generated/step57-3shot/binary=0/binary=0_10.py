
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 3, 1, stride=1, padding=0)
        self.conv2 = torch.nn.Conv2d(3, 4, 1, stride=1, padding=0)
    def forward(self, x1, other):
        v1 = self.conv(x1)
        v2 = self.conv2(x1)
        v3 = v1 + other
        return v3
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
other = 1
