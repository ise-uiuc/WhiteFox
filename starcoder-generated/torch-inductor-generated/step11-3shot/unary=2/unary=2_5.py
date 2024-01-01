
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.deconv = torch.nn.ConvTranspose2d(16, 128, 3, stride=2, padding=3, output_padding=1)
        self.bilinear = torch.nn.UpsamplingBilinear2d(scale_factor=2)
    def forward(self, x):
        x = self.deconv(x)
        return self.bilinear(x)
# Inputs to the model
x1 = torch.randn(2, 16, 18, 38)
