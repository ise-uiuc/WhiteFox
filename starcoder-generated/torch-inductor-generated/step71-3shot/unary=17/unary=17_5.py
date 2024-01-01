
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.ConvTranspose2d(3, 16, 3, stride=1, padding=2)
        self.conv2 = torch.nn.ConvTranspose2d(16, 32, 3, stride=3, padding=1)
        self.conv3 = torch.nn.ConvTranspose2d(32, 64, 3, stride=2, padding=1)
        self.conv4 = torch.nn.ConvTranspose2d(64, 1, 1)
    def forward(self, x1):
        v1 = torch.relu(self.conv1(x1))
        v2 = torch.relu(self.conv2(v1))
        v3 = torch.relu(self.conv3(v2))
        v4 = self.conv4(v3)
        return v4
# Inputs to the model
x1 = torch.randn(1, 3, 30, 30)
