
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(6, 8, 1, stride=1, padding=1)
        self.conv = torch.nn.Conv2d(8, 5, 3, stride=2, padding=4)
    def forward(self, x1):
        v1 = v6 = self.conv_transpose(x1)
        v2 = v6 = self.conv(v6)
        v5 = v6 + 1
        return v6
# Inputs to the model
x1 = torch.randn(1, 6, 64, 64)
