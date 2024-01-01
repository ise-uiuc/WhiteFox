
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 6, 5, stride=(1, 2), bias=False)
        self.conv_t = torch.nn.ConvTranspose2d(6, 6, 5, stride=(1, 2), bias=False)
    def forward(self, x):
        x1 = self.conv(x)
        x2 = self.conv_t(x1)
        x3 = x2 * 0.55
        x4 = torch.pow(x3, 2.11)
        return x4
# Inputs to the model
x = torch.randn(1, 3, 35, 49)
