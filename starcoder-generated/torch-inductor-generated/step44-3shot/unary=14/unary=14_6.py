
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose_11 = torch.nn.ConvTranspose2d(32, 32, 3, stride=1, padding=1)
        self.conv_11 = torch.nn.Conv2d(32, 17, 3, stride=1, padding=1)
    def forward(self, x1):
        v1 = self.conv_transpose_11(x1)
        v2 = torch.sigmoid(v1)
        v3 = v1 * v2
        v4 = self.conv_11(v3)
        return v4
# Inputs to the model
x1 = torch.randn(1, 32, 32, 32)
