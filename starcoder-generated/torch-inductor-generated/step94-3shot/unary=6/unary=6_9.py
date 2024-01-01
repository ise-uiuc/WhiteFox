
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 15, 1, stride=1, padding=1)
        self.padd = torch.nn.ZeroPad2d((1, 0, 1, 2))
        self.bn = torch.nn.Conv2d(15, 10, 3, stride=1, padding=1)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = 3 + v1
        v3 = torch.clamp_min(v2, 0)
        v4 = torch.clamp_max(v3, 6)
        v5 = v1 * v4
        v6 = v5 / 6
        v7 = self.padd(v6)
        v8 = self.bn(v7)
        return v8
# Inputs to the model
x1 = torch.randn(2, 3, 32, 32)
