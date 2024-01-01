
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(133, 8, 1, stride=1, padding=0, output_padding=0, groups=1, bias=False, dilation=1)
    def forward(self, x1):
        t1 = self.conv_transpose(x1)
        t2 = torch.sigmoid(t1)
        t3 = t1 * t2
        return t3
# Inputs to the model
x1 = torch.randn(1, 133, 48, 48)
