
class M(torch.nn.Module):
    def __init__(self, x, y, z):
        super().__init__()
        a = torch.rand(x, y, z) # input tensor
        self.conv_t = torch.nn.ConvTranspose2d(in_channels=x, out_channels=z, kernel_size=z, stride=x, padding=y)
    def forward(self, x1):
        v1 = self.conv_t(x1)
        v2 = torch.sigmoid(v1)
        v3 = v1 * v2
        return v3
# Inputs to the model
M(16, 15, 128)(torch.randn(1, 16, 128, 128))
