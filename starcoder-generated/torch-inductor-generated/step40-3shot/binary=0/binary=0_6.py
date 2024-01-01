
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 7, 1, stride=1, padding=1)
        self.t1 = torch.nn.Sigmoid()
        self.conv2 = torch.nn.Conv2d(7, 2, 1, stride=1, padding=1)
    def forward(self, x1, other=2, padding1=None):
        t1 = self.t1(self.conv(x1))
        if padding1 == None:
            padding1 = torch.randn(1, 5, 5, 5)
        v2 = other * t1
        v1 = v2 + padding1
        return self.conv2(v1)
# Inputs to the model
x1 = torch.randn(1, 3, 10, 10)
x2 = torch.randn(1, 1, 1, 1)
