
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_module1 = torch.nn.Conv2d(39, 33, 1, stride=1, padding=1)
        self.conv_module2 = torch.nn.Conv2d(33, 23, 1, stride=1, padding=1)
    def forward(self, x1):
        v1 = self.conv_module1(x1)
        v2 = self.conv_module2(v1)
        return v2
# Inputs to the model
x1 = torch.randn(1, 39, 64, 64)
