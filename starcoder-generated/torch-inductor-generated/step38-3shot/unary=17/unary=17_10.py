
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.ConvTranspose2d(3, 32, 1, padding=1, stride=1)
        self.conv2 = torch.nn.ConvTranspose2d(32, 16, 1, padding=1, stride=1)
        self.conv3 = torch.nn.ConvTranspose2d(16, 16, 1, padding=1, stride=1)
        self.conv4 = torch.nn.ConvTranspose2d(16, 8, 1, padding=1, stride=1)
        self.conv6 = torch.nn.ConvTranspose2d(8, 1, 1, padding=1, stride=1)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = self.conv2(v1)
        v3 = self.conv3(v2)
        v4 = self.conv4(v3)
        v5 = torch.tanh(v4)
        v6 = self.conv6(v5)
        return v6
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
