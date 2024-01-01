
class Model1(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 3, 1, stride=2, padding=0)
    def forward(self, x1, x2, other1=1, other2=1):
        v1 = self.conv(x1)
        v2 = v1 + other1
        v3 = torch.cat((v2, x2), dim=1)
        return v3

class Model2(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 3, 3, stride=3, padding=4)
    def forward(self, x1, x2, other1=1, other2=1):
        v1 = self.conv(x1)
        v2 = x1 + other2
        v3 = v1 + other1
        return v3
# Inputs to the model
x1 = torch.randn(1, 3, 16, 16)
x2 = torch.randn(1, 3, 14, 14)
