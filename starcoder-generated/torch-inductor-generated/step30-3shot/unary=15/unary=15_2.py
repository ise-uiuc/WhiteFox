
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.deconv1 = torch.nn.ConvTranspose2d(4, 3, 4, stride=2, padding=0, dilation=1)
        self.deconv2 = torch.nn.ConvTranspose2d(3, 4, 3, stride=1, padding=0, dilation=1)
        self.bn1 = torch.nn.BatchNorm2d(4)
    def forward(self, x1):
        v1 = self.deconv1(x1)
        v2 = self.bn1(self.deconv2(v1))
        return v2
# Inputs to the model
x1 = torch.randn(1, 4, 64, 64)
