
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.ConvTranspose2d(in_channels=3, out_channels=2, kernel_size=2, stride=3, padding=1, output_padding=1, groups=2)
    def forward(self, x):
        v1 = self.conv(x)
        v2 = v1 - 1.0
        return v2
# Inputs to the model
x = torch.randn(1, 3, 22, 22)
