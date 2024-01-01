
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.a3 = torch.nn.Conv2d(3, 32, 3, stride=2, padding=0)
        self.a4 = torch.nn.Conv2d(32, 64, 3, stride=2, padding=1)
        self.a5 = torch.nn.Conv2d(64, 128, 3, stride=2, padding=1)
        self.a6 = torch.nn.Conv2d(128, 256, 3, stride=2, padding=1)
    def forward(self, x1, x2, x3, x4, x5):
        x1 = self.a3(x1)
        x2 = self.a4(x2)
        x3 = self.a5(x3)
        x4 = self.a6(x4)
        x6 = torch.cat([x1, x2], 1)
        x7 = torch.cat([x6, x3], 1)
        x8 = torch.cat([x7, x4], 1)
        x9 = torch.cat([x8, x5], 1)
        x10 = x9 + x5
        return x10
# Inputs to the model
self.x1 = torch.randn(2, 3, 64, 64)
self.x2 = torch.randn(2, 3, 31, 31)
self.x3 = torch.randn(2, 3, 15, 15)
self.x4 = torch.randn(2, 3, 7, 7)
self.x5 = torch.randn(2, 3, 3, 3)
