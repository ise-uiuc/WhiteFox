
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.ConvTranspose2d(16, 64, 32, stride=4, padding=10)
        self.conv2 = torch.nn.ConvTranspose2d(64, 128, 16, stride=4, padding=7)
        self.conv3 = torch.nn.ConvTranspose2d(128, 256, 8, stride=4, padding=4)
        self.conv4 = torch.nn.ConvTranspose2d(512, 16, 8, stride=4, padding=4)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = self.conv2(v1)
        v3 = self.conv3(v2)
        v4 = torch.cat([v2, v3], dim=1)
        v5 = self.conv4(v4)
        v6 = v5 + x1
        return v6

# Inputs to the model
x1 = torch.randn(1, 16, 50, 50)
