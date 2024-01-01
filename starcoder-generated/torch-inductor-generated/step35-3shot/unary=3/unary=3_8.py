
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(384, 36, 1, stride=1, padding=0)
        self.conv2 = torch.nn.Conv2d(36, 16, 1, stride=1, padding=0)
        self.conv3 = torch.nn.Conv2d(16, 9, 1, stride=1, padding=0)
        self.conv4 = torch.nn.Conv2d(9, 4, 1, stride=1, padding=0)
        self.conv5 = torch.nn.ConvTranspose2d(4, 32, 1, stride=1, padding=0)
        self.conv6 = torch.nn.ConvTranspose2d(32, 24, 1, stride=1, padding=0)
        self.conv7 = torch.nn.ConvTranspose2d(24, 20, 1, stride=1, padding=0)
        self.conv8 = torch.nn.ConvTranspose2d(20, 16, 1, stride=1, padding=0)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = self.conv2(v1)
        v3 = self.conv3(v2)
        v4 = self.conv4(v3)
        v5 = self.conv5(v4)
        v6 = self.conv6(v5)
        v7 = self.conv7(v6)
        v8 = self.conv8(v7)
        return v8
# Inputs to the model
x1 = torch.randn(1, 384, 32, 128)
