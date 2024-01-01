
class Model3(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 1, stride=1)
        self.conv_transpose = torch.nn.ConvTranspose2d(8, 8, 1, stride=1, padding=0)
    def forward(self, x1):
        x2 = self.conv(x1)
        x3 = self.conv_transpose(x2)
        return x2 + x3
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
