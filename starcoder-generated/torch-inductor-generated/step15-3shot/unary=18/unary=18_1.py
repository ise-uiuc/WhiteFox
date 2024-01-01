
class model(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(model, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(out_channels),
            nn.Sigmoid(),
        )
        self.conv = torch.nn.Conv2d(in_channels, out_channels=128, kernel_size=1, stride=1, padding=1)
    def forward(self, x):
        v1 = self.conv(x)
        v2 = self.conv_block(v1)
        return v2

in_channels = 3
out_channels = 128
kernel_size = 1 
stride = 1 
padding = 1 
# Inputs to the model
x = torch.randn(1, 3, 64, 64)
