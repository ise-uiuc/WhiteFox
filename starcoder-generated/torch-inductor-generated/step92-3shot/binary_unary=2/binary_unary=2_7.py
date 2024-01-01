
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Pointwise convolution with kernel size 3x3x3
        self.conv1 = torch.nn.Conv2d(10, 10, 3, stride=1, padding=1)
        # Conv2DTranspose with kernel size 3x3x3
        self.deconv1 = torch.nn.ConvTranspose2d(10, 10, 3, stride=1, padding=1)
        # AveragePooling with kernel size 3x3x3
        self.avgpool = torch.nn.AvgPool2d(3)
    def forward(self, x):
        t = self.conv1(x)
        u = torch.floor(t)
        v = torch.clamp(t - u, 0., 1.)
        w = self.deconv1(v)
        y = torch.ones_like(w)
        z = torch.ones_like(y)
        a = self.avgpool(z)
        return x

# Inputs to the model
x1 = torch.randn(1, 10, 10, 10)
