
class Model(torch.nn.Module):
    def __init__(self, in_channels = 3, out_channels = 3, kernel_size = 1):
        super().__init__()
        self.conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size)
        self.sigmoid = torch.nn.Sigmoid()
    def forward(self, x1):
        v1 = self.conv(x1)
        return v1
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
