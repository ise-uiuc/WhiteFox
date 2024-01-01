
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.ConvTranspose2d(3, 8, 3, stride=2, padding=1, groups=4)
        self.conv1 = torch.nn.ConvTranspose2d(8, 8, 3, stride=2, padding=1, groups=4)
        self.conv2 = torch.nn.ConvTranspose2d(8, 4, 3, stride=1, padding=1, groups=4)
        self.conv3 = torch.nn.ConvTranspose2d(4, 4, 3, stride=1, padding=1, groups=2)
        self.conv4 = torch.nn.ConvTranspose2d(4, 1, 1, stride=1, padding=0, groups=4)
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
        return v9
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
