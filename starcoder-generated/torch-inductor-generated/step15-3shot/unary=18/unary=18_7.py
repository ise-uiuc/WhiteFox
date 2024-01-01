
class Model(nn.Module):
    def __init__(self, in_channels, kernel_size, stride, output_features):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size, stride)
        self.conv2 = nn.Conv2d(16, 8, kernel_size, stride)
        self.conv3 = nn.Conv2d(8, 4, kernel_size, stride)
        self.conv4 = nn.Conv2d(4, output_features, kernel_size, stride)
    def forward(self, x):
        v1 = self.conv1(x)
        v2 = torch.sigmoid(v1)
        v3 = self.conv2(v2)
        v4 = torch.sigmoid(v3)
        v5 = self.conv3(v4)
        v6 = torch.sigmoid(v5)
        v7 = self.conv4(v6)
        v8 = torch.sigmoid(v7)
        return v8
in_channels = 64
kernel_size = 2
stride = 2
# Inputs to the model
x = torch.randn(1, 3, 64, 64)
