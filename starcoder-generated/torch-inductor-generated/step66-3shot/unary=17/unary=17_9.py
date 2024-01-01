
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.ConvTranspose2d(3, 128, kernel_size=3, stride=4, padding=0)
        self.conv2 = torch.nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1)
        self.conv3 = torch.nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1)
        self.conv4 = torch.nn.ConvTranspose2d(32, 1, 3, stride=1, padding=0)
    def forward(self, x1):
        v2 = self.conv1(x1)
        v3 = torch.relu(v2)
        v4 = self.conv2(v3)
        v5 = torch.sigmoid(v4)
        v6 = self.conv3(v5)
        v7 = torch.relu(v6)
        out1 = self.conv4(v7)
        return out1
# Inputs to the model
x1 = torch.randn(1, 3, 512, 512)
