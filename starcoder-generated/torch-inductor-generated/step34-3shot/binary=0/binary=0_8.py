
class Model(torch.nn.Sequential):
    def __init__(self):
        super().__init__()
        self.add_module('conv0', torch.nn.Conv2d(3, 8, 1, stride=1, padding=2))
    def forward(self, x1, other=None):
        v1 = self.conv0(x1)
        v2 = v1 + other
        return v2
# Inputs to the model
x1 = torch.randn(5, 3, 64, 64)
