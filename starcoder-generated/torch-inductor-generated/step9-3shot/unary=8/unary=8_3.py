
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(8, 39, 1)
    def forward(self, x2):
        return self.conv(x2)
x2 = torch.randn(2, 8, 51, 61)
y2 = torch.randn(2, 39, 51, 61)

