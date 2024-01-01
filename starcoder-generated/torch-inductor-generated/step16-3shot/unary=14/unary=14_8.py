
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 6, 5, stride=1, padding=0)
        self.transposeconv = torch.nn.ConvTranspose2d(6, 9, 7, stride=1, padding=0)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = self.transposeconv(v1)
        v3 = torch.sigmoid(v2)
        v4 = v2 * v3
        return v4
# Inputs to the model
x1 = torch.randn(1, 3, 24, 24)
