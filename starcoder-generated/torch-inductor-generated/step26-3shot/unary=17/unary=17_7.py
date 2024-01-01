
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.ConvTranspose2d(3, 128, 16, stride=1, padding=8)
        self.conv1 = torch.nn.ConvTranspose2d(128, 64, 16, stride=2)
        self.conv2 = torch.nn.ConvTranspose2d(64, 32, 16, stride=1, padding=1)
        self.conv3 = torch.nn.ConvTranspose2d(32, 32, 1, padding=1, stride=1)
        self.conv4 = torch.nn.ConvTranspose2d(32, 16, 8, padding=4, stride=1)
        self.conv5 = torch.nn.ConvTranspose2d(16, 1, 2, padding=1, stride=1)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = torch.relu(v1)
        v3 = self.conv1(v2)
        v4 = torch.relu(v3)
        v5 = self.conv2(v4)
        v6 = torch.relu(v5)
        v7 = self.conv3(v6)
        v8 = self.conv4(v7)
        v9 = self.conv5(v8)
        return v9
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
