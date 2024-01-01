
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.tconv = torch.nn.ConvTranspose2d(3, 3, 3)
        self.conv = torch.nn.Conv2d(3, 3, 3, padding=1)
    def forward(self, x1):
        v1 = self.tconv(x1)
        v2 = torch.tanh(v1)
        v3 = self.conv(v2)
        v4 = torch.sigmoid(v3)
        return v4
# Inputs to the model
x1 = torch.randn(1, 3, 256, 256)
