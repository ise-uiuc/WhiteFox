
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 4, (1, 4))
        self.conv2 = nn.Conv2d(4, 8, 1)
        self.conv3 = nn.Conv2d(8, 16, 3, padding=1)
    def forward(self, x):
        n1 = self.conv(x)
        n2 = self.conv2(n1)
        n3 = self.conv3(n2)
        return n3
