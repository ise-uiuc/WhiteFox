
class MyModule(torch.nn.Module):
    def __init__(self):
        super(MyModule, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 32, 3, stride=2, padding=1)
        self.conv2 = torch.nn.ConvTranspose2d(32, 64, 3, stride=2)
        self.conv3 = torch.nn.ConvTranspose2d(64, 128, 3, stride=2, padding=1)
        self.conv4 = torch.nn.ConvTranspose2d(128, 64, 3, stride=2)
        self.conv5 = torch.nn.ConvTranspose2d(64, 64, 3, stride=2, padding=1)
    def forward(self, x):
        v1 = self.conv1(x)
        v2 = F.relu(v1)
        v3 = self.conv2(v2)
        v4 = F.relu(v3)
        v5 = self.conv3(v4)
        v6 = F.relu(v5)
        v7 = self.conv4(v6)
        v8 = F.relu(v7)
        v9 = self.conv5(v8)
        v10 = F.relu(v9)
        return v10
# Inputs to the model
x = torch.randn(1, 3, 48, 48)
