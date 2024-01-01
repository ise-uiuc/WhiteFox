
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.ConvTranspose2d(256, 256, (3, 1), (2, 2), padding=(2, 1))
        self.conv1 = torch.nn.ConvTranspose2d(128, 128, (3, 3), (1, 1), (2, 2), padding=(2, 0))
        self.conv2 = torch.nn.Conv2d(256, 192, (3, 1), (2, 1), (1, 1), (1, 1))
        self.conv3 = torch.nn.Conv2d(128, 64, (3, 3), (1, 1), (1, 1), (1, 1))
        self.conv4 = torch.nn.ConvTranspose2d(192, 128, (3, 3), (2, 1), (2, 1), padding=(2, 0))
        self.conv5 = torch.nn.ConvTranspose2d(64, 3, (3, 3), (2, 1), (1, 2), padding=(1, 2))
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = torch.relu(v1)
        v3 = self.conv1(v2)
        v4 = torch.relu(v3)
        v5 = self.conv2(v4)
        v6 = self.conv3(v5)
        v7 = torch.relu(v6)
        v8 = self.conv4(v7)
        v9 = self.conv5(v8)
        return v9
# Inputs to the model
x1 = torch.randn(1, 256, 14, 16)
