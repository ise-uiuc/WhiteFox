
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv2d_3 = torch.nn.Conv2d(3, 3, 1, stride=1, padding=0)
        self.conv_transpose_1 = torch.nn.ConvTranspose2d(3, 3, 1, stride=1, padding=0)
    def forward(self, x1):
        v1 = self.conv2d_3(x1)
        v2 = torch.sigmoid(v1)
        v3 = self.conv_transpose_1(v2)
        v4 = torch.nn.functional.interpolate(v3, scale_factor=3.0, mode='nearest')
        v5 = v4 * v1
        v6 = v1 + v5
        return v6
# Inputs to the model
x1 = torch.randn(1, 3, 28, 28)
