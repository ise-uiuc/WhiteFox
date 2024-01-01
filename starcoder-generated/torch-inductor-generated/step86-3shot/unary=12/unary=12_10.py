
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.sigmoid = torch.nn.Sigmoid()
        self.t1 = torch.nn.Conv2d(3, 2, 1, stride=1, padding=1)
        self.t2 = torch.nn.Conv2d(2, 2, 3, stride=1, padding=1)
    def forward(self, x1):
        v1 = self.t1(x1)
        v2 = self.sigmoid(v1)
        v3 = self.t2(v2)
        v4 = self.sigmoid(v3)
        v5 = v3 * v4
        return v5
x1 = torch.randn(1, 3, 64, 64)
