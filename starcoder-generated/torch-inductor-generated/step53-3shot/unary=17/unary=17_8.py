
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.ConvTranspose2d(3, 32, 3, stride=2, padding=1)
        self.conv1 = torch.nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1)
        self.conv2 = torch.nn.ConvTranspose2d(16, 8, 3, stride=2, padding=1)
        self.conv3 = torch.nn.ConvTranspose2d(8, 4, 3, stride=2, padding=1)
        self.conv4 = torch.nn.ConvTranspose2d(4, 2, 3, stride=2, padding=1)
        self.conv5 = torch.nn.ConvTranspose2d(2, 1, 3, stride=2, padding=1)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = self.conv1(v1)
        v3 = self.conv2(v2)
        v4 = self.conv3(v3)
        v5 = self.conv4(v4)
        v6 = self.conv5(v5)
        return v6
# Inputs to the model
x1 = torch.randn(1, 3, 32, 32)
