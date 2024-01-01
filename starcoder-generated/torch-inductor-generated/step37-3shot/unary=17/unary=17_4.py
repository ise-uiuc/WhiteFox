
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.ConvTranspose2d(16, 16, 3, padding=1)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = torch.sigmoid(v1)
        v3 = torch.nn.functional.interpolate(v2, None, None, 1.5, 'nearest')
        return v3
# Inputs to the model
x1 = torch.randn(1, 16, 64, 64)
