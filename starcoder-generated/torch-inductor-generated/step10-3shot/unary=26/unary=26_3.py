
class CustomModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        x2 = self.conv_transpose(x)
        x3 = x2 > 0
        x4 = x2 * self.negative_slope
        x5 = torch.where(x3, x2, x4)
        return x5
class Model(torch.nn.Module):
    def __init__(self, negative_slope=0.25):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(3, 8, 1, stride=1, padding=1)
        self.negative_slope = negative_slope
        self.custom_module = CustomModule()
    def forward(self, x1):
        x2 = x1 + torch.ones_like(x1)
        x3 = self.conv_transpose(x2)
        x4 = self.custom_module(x3)
        return x4
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
