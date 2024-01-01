
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Sequential(torch.nn.ConvTranspose2d(7, 3, 3, padding=1, stride=2, output_padding=1, dilation=1, groups=1, bias=False), torch.nn.ReLU(inplace=False))
    def forward(self, x):
        v1 = self.conv(x)
        return v1
# Inputs to the model
x = torch.randn(1, 7, 16, 16)
