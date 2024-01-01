
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
    def forward(self, x1_3):
        v1 = self.conv(x1_3)
        v2 = v1 - 13.0
        return v2
x1_3 = torch.randn(8, 3, 64, 64)
