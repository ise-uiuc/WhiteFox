
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.transposeconv2d = torch.nn.ConvTranspose2d(8, 8, stride=1, kernel_size=1, padding=1)
    def forward(self, x1):
        v1 = self.transposeconv2d(x1)
        v2 = torch.sigmoid(v1)
        v3 = v1 * v2
        return v3
# Inputs to the model
x1 = torch.randn(1, 8, 8, 64)
