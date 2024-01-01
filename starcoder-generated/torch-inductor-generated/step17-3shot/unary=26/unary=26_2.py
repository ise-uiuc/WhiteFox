
class Model(torch.nn.Module):
    def __init__(self, n_channels):
        super().__init__()
        self.conv_t1 = torch.nn.ConvTranspose2d(n_channels, 256, kernel_size=(4, 8))
        self.conv_t2 = torch.nn.ConvTranspose2d(256, n_channels, kernel_size=(3, 8))
    def forward(self, x2):
        v1 = self.conv_t1(x2)
        v2 = v1 > 0
        v3 = v1 * 0.25
        v4 = torch.where(v2, v1, v3)
        v5 = self.conv_t2(v4)
        return v5
n_channels = 3
# Inputs to the model
x2 = torch.randn(8, n_channels, 8, 8)
