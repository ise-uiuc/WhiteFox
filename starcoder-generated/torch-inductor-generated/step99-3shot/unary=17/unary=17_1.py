
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.ConvTranspose2d(1, 16, 1)
        self.conv2 = torch.nn.ConvTranspose2d(16, 32, 1)
        self.conv3 = torch.nn.ConvTranspose2d(32, 3, 1)
    def forward(self, x1):
        y = self.conv1(x1)
        z = self.conv2(y)
        t = self.conv3(z)
        return t
# Inputs to the model
x1 = torch.randn(1, 1, 10, 10)
