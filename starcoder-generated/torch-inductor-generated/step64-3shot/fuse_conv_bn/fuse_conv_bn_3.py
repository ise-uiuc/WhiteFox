
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        b, c, h, w = 3, 4, 3, 5
        self.conv = torch.nn.Conv2d(c, c, 3, padding=1)
        self.bn = torch.nn.BatchNorm2d(c, affine=False)
        self.pool = torch.nn.MaxPool2d(h)
    def forward(self, img):
        o1 = self.pool(self.bn(self.conv(img)))
        o2 = self.bn(self.conv(o1))
        return o2
    
# Inputs to the model
img = torch.randn(b, c, h, w)
