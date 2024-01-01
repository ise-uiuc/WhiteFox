
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 8, 7, 1)
        self.conv2 = torch.nn.ConvTranspose2d(8, 32, 6, 1)
        self.conv3 = torch.nn.ConvTranspose2d(32, 64, 3, 1)
        self.conv4 = torch.nn.ConvTranspose2d(64, 128, 6, 1)
        self.conv5 = torch.nn.ConvTranspose2d(128, 256, 7, 1)
        self.conv6 = torch.nn.ConvTranspose2d(256, 2, 3, 1)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = torch.relu(v1)
        v3 = torch.relu(v2)
        v4 = torch.relu(v3)
        v5 = self.conv2(v4)
        v6 = torch.relu(v5)
        v7 = torch.relu(v6)
        v8 = self.conv3(v7)
        v9 = torch.relu(v8)
        v10 = torch.relu(v9)
        v11 = self.conv4(v10)
        v12 = torch.relu(v11)
        v13 = torch.relu(v12)
        v14 = self.conv5(v13)
        v15 = torch.relu(v14)
        v16 = torch.relu(v15)
        v17 = self.conv6(v16)
        return v17
# Inputs to the model
x1 = torch.randn(1, 1, 32, 32)
