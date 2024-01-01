
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 4, 3, stride=2)
        self.convt1 = torch.nn.ConvTranspose2d(4, 8, 3, stride=2)
        self.leaky_relu1 = torch.nn.LeakyReLU()
        self.convt2 = torch.nn.ConvTranspose2d(8, 4, 5, stride=1)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = self.convt1(v1)
        v3 = self.leaky_relu1(v2)
        v4 = self.convt2(v3)
        return v4
# Inputs to the model
x1 = torch.randn(1, 1, 27, 36)
