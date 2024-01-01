
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(in_channels=1, out_channels=1, kernel_size=2, stride=2, padding=0)
        self.conv = torch.nn.Conv2d(in_channels=1, out_channels=1, kernel_size=2, stride=1, padding=1)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = self.conv_transpose(v1)
        return v2
# Inputs to the model
x1 = torch.randn(1, 1, 28, 28)
