
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.maxpool2d_1 = torch.nn.MaxPool2d(3, stride=3, padding=3)
        self.maxpool2d_2 = torch.nn.MaxPool2d(3, stride=1, padding=0, dilation=1, ceil_mode=False)
        self.conv_transpose = torch.nn.ConvTranspose2d(3, 3, 3, stride=1, padding=0, output_padding=0, groups=1, dilation=1)
    def forward(self, x1, x2):
        v1 = self.maxpool2d_1(x1)
        v2 = self.maxpool2d_2(x2)
        v3 = torch.max(v1, v2)
        v4 = torch.sigmoid(v3)
        v5 = self.conv_transpose(v4)
        v6 = v5 * v3
        return v6
# Inputs to the model
x1 = torch.randn(1, 3, 256, 256)
x2 = torch.randn(1, 3, 256, 256)
