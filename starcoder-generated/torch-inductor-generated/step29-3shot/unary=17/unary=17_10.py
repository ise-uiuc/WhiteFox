
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.up = torch.nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.conv = torch.nn.ConvTranspose2d(512, 128, 1, stride=1, padding=0)
        self.conv1 = torch.nn.ConvTranspose2d(256, 64, 1, stride=1, padding=0)
        self.fc = torch.nn.ConvTranspose2d(64, 32, 1, stride=1, padding=0)
        self.conv2 = torch.nn.ConvTranspose2d(32, 1, 1, stride=1, padding=0)
    def forward(self, x1):
        v11 = self.up(x1)
        v11 = F.relu(v11)
        v12 = self.conv(x1)
        v12 = F.relu(v12)
        v13 = torch.add(v11, v12)
        v13 = self.conv1(v13)
        v13 = F.relu(v13)
        v15 = self.fc(v13)
        v15 = F.relu(v15)
        v16 = self.conv2(v15)
        v16 = torch.sigmoid(v16)
        return v16
# Inputs to the model
x1 = torch.randn(1, 512, 2, 2)
