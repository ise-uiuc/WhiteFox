
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.ConvTranspose2d(3, 3, kernel_size=3, stride=2, bias=True, dilation=2, padding=1)
        self.conv2 = torch.nn.ConvTranspose2d(1, 1, 1, stride=1, padding=0, output_padding=0, dilation=1, groups=1)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = self.conv2(v1)
        v3 = torch.sigmoid(v2)
        return v3


# Inputs to the model
x1 = torch.randn(1, 3, 76, 25)
