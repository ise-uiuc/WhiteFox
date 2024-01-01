
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose0 = torch.nn.ConvTranspose2d(3, 1, 7)
        self.conv0 = torch.nn.Conv2d(1, 1, 3)
        self.conv_transpose1 = torch.nn.ConvTranspose2d(1, 1, 3, stride=(2, 2))
    def forward(self, x1):
        v1 = self.conv_transpose0(x1)
        v2 = self.conv0(v1)
        v3 = self.conv_transpose1(v2)
        return v3
# Inputs to the model
x1 = torch.randn(1, 3, 32, 32)
