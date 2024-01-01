
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.ConvTranspose2d(1, 16, 1, padding=0, stride=1)
        self.conv1 = torch.nn.ConvTranspose2d(16, 32, 3, padding=1, stride=1)
        self.conv2 = torch.nn.ConvTranspose2d(32, 32, 6, padding=4, stride=1)
        self.conv3 = torch.nn.ConvTranspose2d(32, 64, 3, padding=1, stride=2)
        self.conv4 = torch.nn.ConvTranspose2d(64, 1, 2, padding=0, stride=1)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = torch.relu(v1)
        v3 = self.conv1(v2)
        v4 = torch.sigmoid(v3)
        v5 = self.conv2(v4)
        v6 = torch.tanh(v5)
        v7 = self.conv3(v6)
        v8 = self.conv4(v7)
        return v8
# Inputs to the model
x1 = torch.randn(1, 1, 64, 64)
# Model begins