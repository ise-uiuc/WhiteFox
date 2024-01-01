
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(56, 64, 3, padding=1)
        # The number of groups should be the maximum of the group size and the number of channels
        # in this case, it should be 64
        self.conv = torch.nn.Conv2d(64, 64, 1, groups=64)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = self.conv_transpose(v1)
        v3 = torch.sigmoid(v2)
        return v3
# Inputs to the model
x1 = torch.randn(1, 56, 56, 56)
