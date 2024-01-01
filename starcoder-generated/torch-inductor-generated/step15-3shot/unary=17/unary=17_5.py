
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.ConvTranspose2d(1, 1, 1)
    def forward(self, x1):
        v1 = self.conv(x1)
        return v1
# Inputs to the model
x1 = torch.randn(1, 1, 1017, 8)
