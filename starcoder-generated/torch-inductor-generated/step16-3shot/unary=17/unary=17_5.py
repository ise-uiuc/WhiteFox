
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.ConvTranspose2d(16, 32, 2, stride=1, padding=1)
        self.conv2 = torch.nn.ConvTranspose2d(32, 64, 2, stride=1, padding=1)
        self.conv3 = torch.nn.ConvTranspose2d(64, 64, (2, 5), stride=1, padding=(1, 1))
        self.conv4 = torch.nn.ConvTranspose2d(64, 64, (2, 5), stride=2, padding=(1, 2))
        self.conv5 = torch.nn.ConvTranspose2d(64, 64, (4, 3), stride=2, padding=(2, 3), dilation=2)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v1 = self.conv2(v1)
        v1 = self.conv3(v1)
        v1 = self.conv4(v1)
        v1 = self.conv5(v1)
        return v1
# Inputs to the model
x1 = torch.randn(1, 16, 64, 64)
