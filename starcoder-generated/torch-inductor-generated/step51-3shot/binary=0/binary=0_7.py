
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(2, 6, 1, stride=1, padding=1)
    def forward(self, x1, some_parameter):
        v1 = self.conv(x1)
        v2 = v1 + some_parameter
        return v2
# Inputs to the model
x1 = torch.randn(1, 2, 64, 64)
some_parameter = 1
