
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.ConvTranspose2d(3, 16, 6, stride=1)
        self.conv1 = torch.nn.ConvTranspose2d(16, 32, 9, stride=1)
        self.conv2 = torch.nn.ConvTranspose2d(32, 64, 11, stride=1)
        self.conv3 = torch.nn.ConvTranspose2d(64, 128, 14, stride=1)
        self.conv4 = torch.nn.ConvTranspose2d(128, 16, 4, stride=1)
        self.conv5 = torch.nn.ConvTranspose2d(16, 3, 2, stride=1)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = torch.relu(v1)
        v3 = self.conv1(v2)
        v4 = torch.relu(v3)
        v5 = self.conv2(v4)
        v6 = torch.relu(v5)
        v7 = self.conv3(v6)
        v8 = torch.relu(v7)
        v9 = self.conv4(v8)
        v10 = torch.relu(v9)
        v11 = self.conv5(v10)
        return v11
# Inputs to the model
x1 = torch.randn(1, 3, 145, 145)
