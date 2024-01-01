
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(428, 64, (47, 10), stride=1, padding=(46, 9), groups=1, bias=True)
        self.conv = torch.nn.Conv3d(16, 10, (3, 7, 3), stride=(1, 2, 2), padding=(1, 3, 1), dilation=(1, 1, 1), groups=1, bias=True)
    def forward(self, x):
        v1 = self.conv_t(x)
        v2 = self.conv(v1)
        return v2
# Inputs to the model
x = torch.randn(8, 16, 428, 117, 39)
