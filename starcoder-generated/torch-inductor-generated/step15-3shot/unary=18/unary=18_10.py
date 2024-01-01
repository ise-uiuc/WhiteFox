
class model(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size):
        super(model, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=1)
        self.conv2 = nn.Conv2d(16, 8, kernel_size=1)
        self.conv3 = nn.Conv2d(8, 8, kernel_size=1)
        self.conv4 = nn.Conv2d(4, 10, kernel_size=3, padding=1)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = torch.sigmoid(v1)
        v3 = self.conv2(v2)
        v4 = torch.sigmoid(v3)
        v5 = self.conv3(v4)
        v6 = torch.sigmoid(v5)
        v7 = v2 + v6
        v8 = self.conv4(v7)
        v9 = torch.sigmoid(v8)
        return v9
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
