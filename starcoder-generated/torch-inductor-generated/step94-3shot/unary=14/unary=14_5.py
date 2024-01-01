
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose_5 = torch.nn.ConvTranspose2d(128, 128, 1, stride=1, padding=0)
        self.conv_transpose_6 = torch.nn.ConvTranspose2d(128, 2, 3, stride=1, padding=1)
        self.conv_transpose_8 = torch.nn.ConvTranspose2d(2, 2, 2, stride=2, padding=0)
    def forward(self, x1):
        v1 = self.conv_transpose_5(x1)
        v2 = torch.sigmoid(v1)
        v3 = v1 * v2
        v4 = self.conv_transpose_6(v3)
        v5 = torch.sigmoid(v4)
        v6 = v4 * v5
        v7 = self.conv_transpose_8(v6)
        return v7
# Inputs to the model
x1 = torch.randn(1, 128, 12, 12)
