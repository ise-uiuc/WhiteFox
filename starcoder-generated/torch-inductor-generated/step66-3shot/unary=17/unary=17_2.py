
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(128, 32, kernel_size=3, stride=2, padding=1)
        self.conv1 = torch.nn.ConvTranspose2d(32, 256, kernel_size=2, stride=2, padding=0)
        self.conv2 = torch.nn.Conv2d(256, 64, kernel_size=3, stride=2, padding=1)
        self.conv3 = torch.nn.Conv2d(64, 16, kernel_size=3, stride=1, padding=0)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = torch.relu(v1)
        v3 = self.conv1(v2)
        v4 = nn.LeakyReLU(negative_slope=math.sqrt(0.1))(v3)
        v5 = self.conv2(v4)
        v6 = torch.relu(v5)
        v7 = self.conv3(v6)
        v8 = torch.sigmoid(v7)
        return v8
# Inputs to the model
x1 = torch.randn(1, 128, 100, 32)
