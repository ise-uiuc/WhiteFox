
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.ConvTranspose2d(3, 64, 1, stride=2)
        self.conv_transpose = torch.nn.ConvTranspose2d(3, 64, 1, stride=2)
        self.conv_transpose1 = torch.nn.ConvTranspose2d(64, 32, 3, stride=1)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = self.conv_transpose(v1)
        v3 = self.conv_transpose1(v2)
        return v3
# Inputs to the model
x1 = torch.randn(1, 3, 32, 32)
