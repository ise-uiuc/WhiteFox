
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(16, 16, (1, 1))
        self.conv = torch.nn.Conv2d(16, 64, (1, 1))
    def forward(self, x1):
        v1 = self.conv_transpose(x1)
        v2 = torch.identity(v1)
        v3 = self.conv(v2)
        v4 = torch.sigmoid(v3)
        return v4
# Inputs to the model
x1 = torch.randn(1, 16, 101, 101)
