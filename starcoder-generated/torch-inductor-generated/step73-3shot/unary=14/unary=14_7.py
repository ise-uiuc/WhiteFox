
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv2d_4 = torch.nn.Conv2d(225, 275, 1, stride=1, padding=0)
        self.conv_transpose_22 = torch.nn.ConvTranspose2d(275, 41, 1, stride=1, padding=0)
    def forward(self, x1):
        v1 = self.conv2d_4(x1)
        v2 = self.conv_transpose_22(v1)
        v3 = torch.sigmoid(v2)
        v4 = v2 * v3
        return v4
# Inputs to the model
x1 = torch.randn(1, 225, 2, 2)
