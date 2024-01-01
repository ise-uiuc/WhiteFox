
class Model(torch.nn.Sequential):
    def __init__(self):
        super().__init__()
        self.add_module('conv', torch.nn.ConvTranspose2d(1, 8, 3, stride=1))
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = torch.nn.ReLU(v1)
        v3 = v1 + v2
        return x1
# Inputs to the model
x1 = torch.randn(1, 1, 28, 28)
