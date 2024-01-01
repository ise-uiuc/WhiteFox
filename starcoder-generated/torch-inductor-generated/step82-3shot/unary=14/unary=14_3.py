
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose_7_1 = torch.nn.ConvTranspose2d(7, 66, 1, stride=1, padding=0)
        self.conv_transpose_66_1 = torch.nn.ConvTranspose2d(66, 57, 1, stride=1, padding=0)
    def forward(self, x1):
        v1 = self.conv_transpose_7_1(x1)
        v2 = torch.sigmoid(v1)
        v3 = v1 * v2
        v4 = self.conv_transpose_66_1(v3)
        v5 = torch.sigmoid(v4)
        v6 = v4 * v5
        return v6
# Inputs to the model
x1 = torch.randn(1, 7, 640, 640)
