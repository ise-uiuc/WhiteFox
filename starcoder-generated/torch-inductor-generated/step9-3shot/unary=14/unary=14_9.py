
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.transposeconv2d1 = torch.nn.ConvTranspose2d(1, 1, stride=1, kernel_size=2, padding=1)
        self.transposeconv2d2 = torch.nn.ConvTranspose2d(1, 1, stride=1, kernel_size=2, padding=1)
        self.transposeconv2d3 = torch.nn.ConvTranspose2d(1, 1, stride=1, kernel_size=2, padding=1)
        self.transposeconv2d4 = torch.nn.ConvTranspose2d(1, 1, stride=1, kernel_size=2, padding=1)
        self.transposeconv2d5 = torch.nn.ConvTranspose2d(1, 1, stride=1, kernel_size=2, padding=1)
        self.transposeconv2d6 = torch.nn.ConvTranspose2d(1, 1, stride=1, kernel_size=2, padding=1)
    def forward(self, x1):
        v1 = self.transposeconv2d1(x1)
        v2 = self.transposeconv2d2(v1)
        v3 = self.transposeconv2d3(v2)
        v4 = self.transposeconv2d4(v3)
        v5 = self.transposeconv2d5(v4)
        v6 = self.transposeconv2d6(v5)
        v7 = torch.sigmoid(v6)
        v8 = v6 * v7
        return v8
# Inputs to the model
x1 = torch.randn(1, 1, 3137, 5)
