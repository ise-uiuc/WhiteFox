
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.deconv = torch.nn.ConvTranspose2d(16, 3, 4, stride=2, padding=2)
    def forward(self, x1):
        v1 = self.deconv(x1)
        v2 = torch.sigmoid(v1)
        v3 = v1 * v2
        return v3
# Inputs to the model
x1 = torch.randn(1, 16, 64, 64)
