
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose0 = torch.nn.ConvTranspose2d(1, 1, 2, stride=2, padding=0, groups=2)
        self.conv_transpose1 = torch.nn.ConvTranspose2d(1, 1, 2, stride=2, padding=0, dilation=2, groups=2)
    def forward(self, x1):
        v1 = self.conv_transpose0(x1)
        v2 = self.conv_transpose1(v1)
        return v2
# Inputs to the model
x1 = torch.randn(1, 1, 7, 7)
