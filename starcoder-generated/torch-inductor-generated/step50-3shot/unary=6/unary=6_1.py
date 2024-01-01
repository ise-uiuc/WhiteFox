
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 3, 1, stride=1, padding=0)
    def forward(self, x1):
        t1 = self.conv(x1)
        t2 = self.conv(x1)
        return t1*t2
# Inputs to the model
x1 = torch.randn(1, 3, 28, 28)
