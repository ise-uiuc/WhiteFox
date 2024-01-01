
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.ConvTranspose2d(8, 64, 7, stride=2, padding=3)
        self.conv1 = torch.nn.ConvTranspose2d(64, 32, 3, padding=1)
        self.conv2 = torch.nn.ConvTranspose2d(32, 8, 3, padding=1)
        self.conv3 = torch.nn.ConvTranspose2d(8, 1, 7, padding=3)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = torch.relu(v1)
        v3 = self.conv1(v2)
        v4 = torch.relu(v3)
        v5 = self.conv2(v4)
        v6 = torch.relu(v5)
        v7 = self.conv3(v6)
        return v7
# Inputs to the model
x1 = torch.randn(1, 8, 256, 256)
