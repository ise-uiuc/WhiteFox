
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.ConvTranspose2d(3, 16, 2)
    def forward(self, x2):
        v1 = self.conv(x2)
        v2 = v1 - 16692004
        return v2
# Inputs to the model
x2 = torch.randn(3, 3, 16, 16)
