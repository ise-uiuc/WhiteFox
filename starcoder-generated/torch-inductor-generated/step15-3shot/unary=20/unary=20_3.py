
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.tconvs1 = torch.nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=2, stride=2, padding=0, dilation=1, output_padding=0, groups=1, bias=True)
    def forward(self, x1):
        v1 = self.tconvs1(x1)
        v2 = torch.sigmoid(v1)
        return v2
# Inputs to the model
x1 = torch.randn(1, 1024, 64, 64)
