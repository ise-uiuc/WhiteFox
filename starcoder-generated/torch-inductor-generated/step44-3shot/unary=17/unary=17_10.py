
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 5, 5, padding=3)
        self.conv_transpose = torch.nn.ConvTranspose2d(5, 10, 5, padding=4, dilation=1)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = self.conv_transpose(v1)
        return v2
# Inputs to the model
x1 = torch.randn(1, 3, 256, 256)
