
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, stride=2, padding=0, kernel_size=3, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, stride=1, padding=1, kernel_size=3, bias=False)
        self.conv3 = nn.ConvTranspose2d(64, 32, stride=2, padding=0, kernel_size=2, output_padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
    def forward(self, x):
        v1 = self.conv1(x)
        v2 = self.bn1(v1)
        v3 = self.conv2(v1)
        v4 = self.conv3(v3)
        v5 = self.bn2(v3)
        v6 = v4 + v5
        return v6
# Inputs to the model
x = torch.randn(1, 3, 64, 64)
