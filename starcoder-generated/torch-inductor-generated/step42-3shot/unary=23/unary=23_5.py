
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 1, 4, stride=2, padding=0, groups=1)
        self.conv_transpose = torch.nn.ConvTranspose2d(1, 1, 4, stride=3, padding=1, groups=1)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = self.conv(x1)
        v3 = self.conv_transpose(v2)
        v4 = torch.tanh(v3)
        return v3
# Inputs to the model
x1 = torch.randn(1, 1, 22, 22)
