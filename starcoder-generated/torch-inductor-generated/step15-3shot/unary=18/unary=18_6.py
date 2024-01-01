
class model(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(model, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding)
    def forward(self, x):
        v1 = self.conv1(x)
        v2 = nn.Sigmoid(v1)
        v3 = self.conv2(v2)
        v4 = nn.Sigmoid(v3)
        return v4
in_channels = 3
out_channels = 16
kernel_size = 1
stride = 1
padding = 1
# Inputs to the model
x = torch.randn(1, 3, 64, 64)
