    
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.ConvTranspose2d(3, 128, 16, stride=1, padding=8)
        self.conv1 = torch.nn.Conv2d(128, 128, 1, padding=0)
        self.conv2 = torch.nn.ConvTranspose2d(128, 64, 16, stride=2)
        self.conv3 = torch.nn.Conv2d(64, 64, 1, padding=0)
        self.conv4 = torch.nn.ConvTranspose2d(64, 32, 16, stride=1, padding=1)
        self.conv5 = torch.nn.Conv2d(32, 32, 1, padding=0)
        self.conv6 = torch.nn.ConvTranspose2d(32, 32, 16, stride=1, padding=1)
        self.conv7 = torch.nn.Conv2d(32, 32, 1, padding=0)
        self.conv8 = torch.nn.ConvTranspose2d(32, 16, 8, padding=4, stride=1)
        self.conv9 = torch.nn.Conv2d(16, 16, 1, padding=0)
        self.conv10 = torch.nn.ConvTranspose2d(16, 1, 2, padding=1, stride=1)
        self.conv11 = torch.nn.Conv2d(1, 1, 1, padding=0)
        
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = self.conv1(v1)
        v3 = torch.relu(v2)
        v4 = self.conv2(v3)
        v5 = self.conv3(v4)
        v6 = torch.relu(v5)
        v7 = self.conv4(v6)
        v8 = self.conv5(v7)
        v9 = torch.relu(v8)
        v10 = self.conv6(v9)
        v11 = self.conv7(v10)
        v12 = torch.relu(v11)
        v13 = self.conv8(v12)
        v14 = self.conv9(v13)
        v15 = torch.relu(v14)
        v16 = self.conv10(v15)
        v17 = self.conv11(v16)
        return v17
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
