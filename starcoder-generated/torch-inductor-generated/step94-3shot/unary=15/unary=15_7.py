
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 16, 5, stride=1, padding=2)
        self.conv2 = torch.nn.Conv2d(16, 32, 11, stride=1, padding=5)
        self.conv3 = torch.nn.ConvTranspose2d(32, 16, 11, stride=1, padding=5)
        self.conv4 = torch.nn.ConvTranspose2d(16, 8, 7, stride=2, padding=3)
        self.conv5 = torch.nn.ConvTranspose2d(8, 1, 19, stride=2, padding=9)
    def forward(self, x1):
        v1 = torch.relu(self.conv1(x1))
        v2 = torch.relu(self.conv2(v1))
        v3 = torch.relu(self.conv3(v2))
        v4 = torch.relu(self.conv4(v3))
        v5 = torch.relu(self.conv5(v4))
        return torch.sigmoid(v5)
# Inputs to the model
x1 = torch.randn(1, 1, 256, 256)
