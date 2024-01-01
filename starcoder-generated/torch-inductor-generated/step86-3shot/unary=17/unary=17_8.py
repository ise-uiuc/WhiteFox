
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.ConvTranspose2d(16, 16, 1, stride=1, padding=0)
        self.conv1 = torch.nn.ConvTranspose2d(16, 16, 1, stride=1, padding=0)
        self.conv2 = torch.nn.ConvTranspose2d(18, 16, 3, stride=2, padding=1)
        self.conv3 = torch.nn.ConvTranspose2d(16, 64, 3, stride=2, padding=1)
        self.conv4 = torch.nn.ConvTranspose2d(64, 128, 2, stride=2, padding=0)
        self.conv5 = torch.nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1)
        self.conv6 = torch.nn.ConvTranspose2d(64, 128, 3, stride=1, padding=1)
        self.conv7 = torch.nn.ConvTranspose2d(128, 64, 3, stride=1, padding=0)
        self.conv8 = torch.nn.ConvTranspose2d(64, 128, 2, stride=1, padding=0)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = torch.relu(v1)
        v3 = self.conv1(v2)
        v4 = torch.relu(v3)
        v5 = self.conv2(v4)
        v6 = torch.relu(v5)
        v7 = self.conv3(v6)
        v8 = torch.relu(v7)
        v9 = v8.transpose(3, 2)
        v10 = self.conv4(v9)
        v11 = torch.relu(v10)
        v12 = v11.transpose(3, 2)
        v13 = self.conv5(v12)
        v14 = torch.relu(v13)
        v15 = v14.transpose(3, 2)
        v16 = self.conv6(v15)
        v17 = torch.relu(v16)
        v18 = v17.transpose(3, 2)
        v19 = self.conv7(v18)
        v20 = torch.relu(v19)
        v21 = v20.transpose(3, 2)
        v22 = self.conv8(v21)
        v23 = torch.relu(v22)
        v24 = torch.sigmoid(v23)
        return v24
# Inputs to the model
x1 = torch.randn(1, 16, 32, 32)
