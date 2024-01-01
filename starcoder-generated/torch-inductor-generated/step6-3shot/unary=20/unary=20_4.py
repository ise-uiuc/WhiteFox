
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose0 = torch.nn.ConvTranspose2d(1, 2, 3, stride=2, padding=0)
        self.conv_transpose1 = torch.nn.ConvTranspose2d(2, 1, 3, stride=2, padding=2, output_padding=0, dilation=2, groups=1)
    def forward(self, x1):
        v1 = self.conv_transpose0(x1)
        v2 = self.conv_transpose1(v1)
        v3 = torch.sigmoid(v2)
        return v3
# Inputs to the model
x1 = torch.randn(1, 1, 16, 16)
