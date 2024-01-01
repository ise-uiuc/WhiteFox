
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dconv1 = torch.nn.ConvTranspose2d(3, 27, 9, stride=1, padding=1)
        self.conv = torch.nn.Conv2d(3, 3, 1, stride=1, padding=1)
    def forward(self, x1):
        x2 = self.dconv1(x1)
        x3 = self.conv(x2)
        return x3
# Inputs to the model
x1 = torch.randn(1, 3, 9, 9)
