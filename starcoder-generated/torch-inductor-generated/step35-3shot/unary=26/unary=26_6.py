
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.ConvTranspose2d(3, 16, 2, stride=1, padding=1)
        self.conv_t = torch.nn.ConvTranspose2d(16, 16, 2, stride=1, bias=False)
        self.bn = torch.nn.BatchNorm2d(16)
    def forward(self, x):
        x1 = self.conv(x)
        x2 = self.conv_t(x1)
        x3 = self.bn(x2)
        return x3
# Inputs to the model
x = torch.randn(2, 3, 32, 32)
