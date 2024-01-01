
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.prelu = torch.nn.PReLU()
        self.conv = torch.nn.Conv2d(16, 8, 1, stride=1, padding=0)
        self.sigmoid = torch.nn.Sigmoid()
    def forward(self, x1):
        v1 = self.prelu(x1)
        v2 = self.conv(v1)
        v3 = self.sigmoid(v2)
        return v3
x1 = torch.randn(1, 16, 32, 32)
