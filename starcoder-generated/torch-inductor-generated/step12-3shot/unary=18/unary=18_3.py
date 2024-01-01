
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.m0 = torch.nn.Conv2d(3, 8, 1, stride=1, padding=0, bias=False)
        self.m1 = torch.nn.Conv2d(8, 16, 1, stride=1, padding=0, bias=False)
        self.m2 = torch.nn.Conv2d(16, 32, 1, stride=1, padding=0, bias=False)
        self.m3 = torch.nn.Conv2d(32, 1, 1, stride=1, padding=0, bias=False)
    def forward(self, x):
        y0 = F.relu(self.m0(x))
        y1 = F.relu(self.m1(y0))
        y2 = F.relu(self.m2(y1))
        y3 = self.m3(y2)
        return y3
# Inputs to the model
x, = torch.randn(1, 3, 64, 64)
