
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv2dtranspose1 = torch.nn.ConvTranspose2d(8, 7, 2, stride=2)
    def forward(self, x1):
        v1 = self.conv2dtranspose1(x1)
        v3 = v1 * v1
        return v3
# Inputs to the model
x1 = torch.randn(1, 8, 56, 56)
