
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
    def forward(self, x):
        v = torch.unsqueeze(x, 2)
        y = x.roll(10000, 1, 2)
        v1 = self.conv(v)
        v2 = v1 - y
        return v2
# Inputs to the model
x = torch.randn(1, 3, 64, 64)
