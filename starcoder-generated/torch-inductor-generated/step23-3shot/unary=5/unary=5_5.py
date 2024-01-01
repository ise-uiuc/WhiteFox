
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.ConvTranspose2d = torch.nn.ConvTranspose2d(2, 2, 3)
        self.AvgPool2d = torch.nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
    def forward(self, x1):
        v1 = self.ConvTranspose2d(x1)
        v2 = self.AvgPool2d(v1)
        v3 = v2 * 0.5
        v4 = v2 * 0.7071067811865476
        v5 = torch.erf(v4)
        v6 = v5 + 1
        v7 = v3 * v6
        return v7
# Inputs to the model
x1 = torch.randn(1, 2, 5, 5)
