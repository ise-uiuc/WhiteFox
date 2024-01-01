
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(3, 5, 6, stride=2)
        self.conv_transpose_2 = torch.nn.ConvTranspose2d(5, 2, 3, stride=2)
        self.conv = torch.nn.Conv2d(2, 6, 5, stride=1)
    def forward(self, x1):
        v1 = self.conv_transpose(x1)
        v2 = self.conv(v1)
        v3 = self.conv_transpose_2(v2)
        return v3
# Inputs to the model
x1 = torch.randn(2, 3, 32, 32)
