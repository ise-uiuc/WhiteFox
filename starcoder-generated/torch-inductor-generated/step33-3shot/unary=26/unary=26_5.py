
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(256, 128, 3, padding=2, output_padding=2, bias=False)
        self.bn = torch.nn.BatchNorm2d(128, affine=False)
        self.act = torch.nn.Hardswish(inplace=False)
    def forward(self, x):
        x1 = self.conv_t(x)
        x2 = self.bn(x1)
        x3 = self.act(x2)
        return x3
# Inputs to the model
x = torch.randn(4, 256, 32, 60)
