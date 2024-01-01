
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_2 = torch.nn.Conv2d(32, 256, 3, stride=1, padding=0)
        self.conv_transpose_1 = torch.nn.ConvTranspose2d(128, 32, 1, stride=1, padding=0)
        self.conv_1 = torch.nn.Conv2d(512, 64, 3, stride=1, padding=0)
    def forward(self, x1):
        v1 = self.conv_2(x1)
        v2 = self.conv_transpose_1(v1)
        v3 = torch.sigmoid(v2)
        v4 = v2 * v3
        v5 = self.conv_1(v4)
        return v5
# Inputs to the model
x1 = torch.randn(1, 512, 64, 64)
