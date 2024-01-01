
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.ConvTranspose2d(2, 4, 3, stride=2, groups=2)
    def forward(self, x1):
        v1 = self.conv(x1)
        return torch.abs(v1)
# Inputs to the model
x1 = torch.randn(2, 2, 4, 4)
