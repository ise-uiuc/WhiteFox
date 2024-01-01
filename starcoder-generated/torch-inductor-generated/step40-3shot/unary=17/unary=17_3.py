
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.ConvTranspose2d(3, 32, 1, stride=1)
        self.conv2 = torch.nn.ConvTranspose2d(32, 64, 2, stride=2)
        self.conv3 = torch.nn.ConvTranspose2d(64, 128, 2, stride=2)
        self.conv4 = torch.nn.ConvTranspose2d(128, 256, 7, stride=1, padding=3)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = F.relu(v1)
        v3 = self.conv2(v2)
        v4 = F.relu(v3)
        v5 = self.conv3(v4)
        v6 = F.relu(v5)
        v7 = self.conv4(v6)
        return v7
# Inputs to the model
x1 = torch.randn(1, 3, 224, 224)
