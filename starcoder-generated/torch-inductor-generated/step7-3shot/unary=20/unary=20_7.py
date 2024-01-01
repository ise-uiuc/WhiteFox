
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv2d_1 = torch.nn.utils.spectral_norm(torch.nn.Conv2d(48, 64, 5, stride=1, padding=2, groups=1, bias=True))
        self.conv2d_transpose_1 = torch.nn.utils.spectral_norm(torch.nn.ConvTranspose2d(64, 64, 5, stride=2, padding=0, output_padding=0, groups=1, bias=True))
    def forward(self, x1):
        v1 = self.conv2d_1(x1)
        v2 = self.conv2d_transpose_1(v1)
        return v2
# Inputs to the model
x1 = torch.randn(1, 48, 32, 32)
