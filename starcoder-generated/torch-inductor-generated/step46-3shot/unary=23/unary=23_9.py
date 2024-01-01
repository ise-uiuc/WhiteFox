
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.ConvTranspose2d(4, 4, 3, stride=2)
    def forward(self, x_d):
        x = self.conv(x_d)
        x = torch.atan(x)
        return x
# Inputs to the model
x1 = torch.randn(1, 4, 173, 173)
