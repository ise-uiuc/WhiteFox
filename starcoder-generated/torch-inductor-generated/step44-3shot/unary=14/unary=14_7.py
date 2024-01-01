
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_14 = torch.nn.Conv2d(999, 999, 1, stride=1, padding=0)
        self.conv_transpose_15 = torch.nn.ConvTranspose2d(999, 999, 1, stride=1, padding=0)
    def forward(self, x1):
        v1 = self.conv_14(x1)
        v2 = torch.sigmoid(v1)
        v3 = v1 * v2
        v4 = self.conv_transpose_15(v3)
        return v4
# Inputs to the model
x1 = torch.randn(1, 999, 224, 224)
