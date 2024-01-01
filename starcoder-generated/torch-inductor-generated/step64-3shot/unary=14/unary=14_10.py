
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose_1 = torch.nn.ConvTranspose2d(128, 4, 1, stride=1, padding=1)
        self.conv_1 = torch.nn.Conv2d(4, 1, 1, stride=1, padding=1)
    def forward(self, x1):
        v1 = self.conv_transpose_1(x1)
        v2 = torch.sigmoid(v1)
        v3 = v1 * v2
        v4 = self.conv_1(v3)
        return v4
# Inputs to the model
x1 = torch.randn(1, 128, 16, 16)
