
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=True, padding_mode="zeros")
    def forward(self, x):
        y = self.conv(x)
        x = torch.cat((y, y), dim=1).tanh()
        return x
# Inputs to the model
x = torch.randn(2, 1, 32, 32)
