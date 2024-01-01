
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(10, 8, 6, bias=False)
        self.conv1x1 = torch.nn.Conv2d(8, 1, 1, bias=False)
        self.conv2x2_t = torch.nn.ConvTranspose2d(1, 1, 2)
    def forward(self, x2):
        y = self.conv_t(x2)
        y = y > 0
        x = self.conv1x1(y)
        d = self.conv2x2_t(x)
        return d
# Inputs to the model
x2 = torch.randn(1, 10, 56, 56)
