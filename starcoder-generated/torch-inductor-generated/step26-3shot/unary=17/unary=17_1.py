
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.ConvTranspose2d(3, 4, 3, stride=2, padding=1)
        self.conv1 = torch.nn.ConvTranspose2d(4, 16, 1)
        self.conv2 = torch.nn.ConvTranspose2d(16, 16, 3, padding=1)
        self.conv3 = torch.nn.ConvTranspose2d(16, 16, 2, padding=1)
        self.conv4 = torch.nn.ConvTranspose2d(16, 16, 1, padding=1)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = self.conv1(v1)
        v3 = torch.relu(v2)
        v4 = self.conv2(v3)
        v5 = torch.relu(v4)
        v6 = self.conv3(v5)
        v7 = self.conv4(v6)
        return v7
# Inputs to the model
x1 = torch.randn(1, 3, 32, 32)
