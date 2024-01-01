
class ConvModel(torch.nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self._conv = torch.nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=2, padding=1)
    def forward(self, x):
        x1 = self._conv(x)
        x2 = self._conv(x)
        x3 = self._conv(x)
        return x1, x2, x3
# Inputs to the model
x1 = torch.randn(1, 56, 64, 64)
