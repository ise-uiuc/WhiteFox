
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.ConvTranspose3d(3, 16, 2, stride=2)
        self.conv1 = torch.nn.ConvTranspose3d(16, 32, 2, stride=2)
        self.conv2 = torch.nn.ConvTranspose3d(32, 1, 2, stride=2)
    def forward(self, x):
        v = self.conv(x)
        v1 = self.conv1(v)
        v2 = self.conv2(v1)
        return v2
# Inputs to the model
x = torch.randn(1, 3, 8, 8, 8)
