
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 3, 1, stride=1, padding=1)
        self.pad2d = torch.nn.ZeroPad2d((1, 1, 0, 1))  # Padding in this case is '1'
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = v1 + 3
        v3 = torch.clamp(v2, 0, 6)
        v4 = v1 * v3
        v5 = v4 / 6
        v6 = self.pad2d(v5)            