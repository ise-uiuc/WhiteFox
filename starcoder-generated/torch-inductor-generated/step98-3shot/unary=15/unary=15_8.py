
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(21, 128, 5, stride=1, padding=2)
        self.conv2 = torch.nn.Conv2d(128, 256, 5, stride=1, padding=2)
        self.conv3 = torch.nn.Conv2d(256, 512, 5, stride=1, padding=2)
        self.conv4 = torch.nn.Conv2d(512, 1024, 1, stride=1, padding=0)
        self.conv5 = torch.nn.ConvTranspose2d(1024, 512, 1, stride=1, padding=0)
        self.conv6 = torch.nn.ConvTranspose2d(512, 256, 1, stride=1, padding=0)
        self.conv7 = torch.nn.ConvTranspose2d(256, 128, 1, stride=1, padding=0)
        self.conv8 = torch.nn.ConvTranspose2d(128, 64, 1, stride=1, padding=0)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = torch.relu(v1)
        v3 = self.conv2(v2)
        v4 = torch.relu(v3)
        v5 = self.conv3(v4)
        v6 = torch.relu(v5)
        v7 = self.conv4(v6)
        v8 = torch.relu(v7)
        v9 = self.conv5(v8)
        v10 = torch.relu(v9)
        v11 = self.conv6(v10)
        v12 = torch.relu(v11)
        v13 = self.conv7(v12)
        v14 = torch.relu(v13)
        v15 = self.conv8(v14)
        v16 = torch.relu(v15)
        return v16
# Inputs to the model
x1 = torch.randn(1, 21, 128, 128)
