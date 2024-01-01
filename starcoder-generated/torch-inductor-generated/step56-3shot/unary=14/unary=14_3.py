
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose_0 = torch.nn.ConvTranspose2d(128, 64, 3, stride=1, padding=1)
        self.conv_transpose_1 = torch.nn.ConvTranspose2d(64, 64, 3, stride=1, padding=1)
    def forward(self, x1):
        v1 = self.conv_transpose_0(x1)
        v2 = self.conv_transpose_1(v1)
        return v2
# Inputs to the model
x1 = torch.randn(1, 128, 256, 256)
