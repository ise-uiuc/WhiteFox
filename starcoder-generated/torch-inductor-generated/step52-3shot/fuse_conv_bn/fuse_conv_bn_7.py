
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 8, 3, stride=1, padding=1, bias=False)
        self.bn_1 = torch.nn.BatchNorm2d(8)
        self.conv2 = torch.nn.Conv2d(3, 8, 1, stride=2, padding=0, bias=False)
        self.bn_2 = torch.nn.BatchNorm2d(8)
    def forward(self, x1):
        y = self.conv1(x1)
        r = self.bn_1(y)
        w = self.conv2(x1)
        p = self.bn_2(w)
        return p
# Inputs to the model
x1 = torch.randn(1, 3, 6, 6)
