
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 7, 16, stride=3, padding=5, dilation=3)
        self.tconv1 = torch.nn.ConvTranspose2d(7, 13, 4, stride=2)
        self.tconv2 = torch.nn.ConvTranspose2d(13, 19, 5, stride=3, padding=4)
    def forward(self, x):
        v1 = self.conv(x)
        v2 = torch.tanh(v1)
        v3 = self.tconv1(v2)
        v4 = torch.tanh(v3)
        return self.tconv2(v4)
# Inputs to the model
x = torch.randn(1, 3, 4, 64)
