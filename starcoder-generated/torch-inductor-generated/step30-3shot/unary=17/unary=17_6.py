
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.ConvTranspose2d(50, 100, 2, stride=2)
        self.conv1 = torch.nn.ConvTranspose2d(100, 25, 2, stride=1)
        self.conv2 = torch.nn.ConvTranspose2d(25, 75, 3, stride=1)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = self.conv1(v1)
        v3 = self.conv2(v2)
        return v3
# Inputs to the model
x1 = torch.randn(1, 1, 140, 1)
