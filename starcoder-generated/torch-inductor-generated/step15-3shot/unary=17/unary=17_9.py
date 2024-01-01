
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.ConvTranspose2d(3, 32, 14)
        self.conv1 = torch.nn.ConvTranspose2d(32, 32, 3)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = self.conv1(v1)
        v3 = torch.sigmoid(v2)
        v4 = torch.tanh(v3)
        return torch.conv2d(v4)
# Inputs to the model
x1 = torch.randn(1, 3, 512, 512)
