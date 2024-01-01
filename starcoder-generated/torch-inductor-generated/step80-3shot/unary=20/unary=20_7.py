
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(18, 163, kernel_size=5, stride=3)
        self.conv = torch.nn.Conv2d(163, 3, kernel_size=3)
    def forward(self, x1):
        v1 = self.conv_t(x1)
        v2 = self.conv(v1)
        v2 = torch.sigmoid(v2)
        return v2
# Inputs to the model
x1 = torch.randn(1, 18, 56, 63)
