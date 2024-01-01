
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.ConvTranspose2d(in_channels=3, out_channels=1, kernel_size=3, groups=1, stride=1, padding=1, bias=False, dilation=1)
    def forward(self, x1):
        v1 = self.conv1(x1)
        return v1
# Inputs to the model
x1 = torch.randn(1, 3, 5, 5)
