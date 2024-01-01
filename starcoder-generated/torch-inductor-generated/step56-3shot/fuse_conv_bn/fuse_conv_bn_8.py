
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv = nn.Conv2d(5, 9, 7, 1, padding=(3, 0))
        self.bn1 = nn.BatchNorm2d(9, momentum=0.8)
        self.bn2 = nn.BatchNorm2d(9)
    def forward(self, x):
        x = self.conv(x)
        a = self.bn1(x)
        b = self.bn2(a)
        return b+x
# Inputs to the model
x = torch.randn(1, 5, 7, 7)
