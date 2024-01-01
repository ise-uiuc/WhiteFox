
class Model2(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 3, 3)
    def forward(self, x1):
        x2 = self.conv(x)
        y = torch.add(x2, x2)
        s = x2.size(1)
        y = torch.split(y, s)
        return y[0] + y[1]
# Input to the model
x1 = torch.randn(1, 3, 4, 4)
