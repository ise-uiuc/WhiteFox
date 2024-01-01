
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.ConvTranspose2d(5, 5, 4, stride=2, padding=1, groups=5, dilation=1, output_padding=0, bias=False)
        self.conv1 = torch.nn.ConvTranspose2d(5, 5, 4, stride=2, padding=1, dilation=1, output_padding=1, bias=False)
        self.conv2 = torch.nn.ConvTranspose2d(5, 5, 4, stride=2, padding=3, dilation=5, output_padding=1, bias=False)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = self.conv1(x1)
        v3 = self.conv2(x1)
        return v1
# Inputs to the model
x1 = torch.randn(1, 5, 128, 128)
