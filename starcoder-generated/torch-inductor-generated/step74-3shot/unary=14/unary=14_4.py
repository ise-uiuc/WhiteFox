
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose_23 = torch.nn.ConvTranspose2d(64, 64, 3, stride=1, padding=1, dilation=1)
        self.conv1 = torch.nn.Conv2d(64, 160, 1, stride=1, padding=0)
    def forward(self, x1):
        v1 = self.conv_transpose_23(x1)
        v2 = self.conv1(v1)
        v3 = torch.sigmoid(v2)
        v4 = v2 * v3
        return v4
# Inputs to the model
x1 = torch.randn(1, 64, 24, 24)
