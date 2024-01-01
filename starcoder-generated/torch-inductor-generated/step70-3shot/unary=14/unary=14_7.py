
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose3 = torch.nn.ConvTranspose2d(150, 3, 2, stride=2, padding=0, dilation=1, groups=1, bias=False)
    def forward(self, x1):
        v1 = self.conv_transpose3(x1)
        v2 = torch.sigmoid(v1)
        v3 = v1 * v2
        return v3
# Inputs to the model
x1 = torch.randn(1, 150, 13, 13)
