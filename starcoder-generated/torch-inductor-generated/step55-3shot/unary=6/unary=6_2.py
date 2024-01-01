
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = torch.nn.AvgPool2d(3, stride=1, padding=1)
        self.conv = torch.nn.Conv2d(5, 3, 1, stride=1, padding=0)
    def forward(self, x1, x2):
        # a branch
        a1 = self.pool(x1)
        a2 = self.conv(a1)
        a3 = 3 + a2
        # b branch
        b1 = self.pool(x2)
        b2 = self.conv(b1)
        b3 = 3 + b2
        return a3 + b3
# Inputs to the model
x1 = torch.randn(1, 5, 224, 288)
x2 = torch.randn(1, 5, 224, 288)
