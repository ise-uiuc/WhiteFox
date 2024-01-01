
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 3, 1, stride=1, padding=1)
    def forward(self, x1):
        t1 = self.conv(x1)
        t2 = torch.add(t1, 3)
        t3 = torch.mul(t2, 6)
        return t3
# Inputs to the model
x1 = torch.randn(3, 3, 64, 64)
