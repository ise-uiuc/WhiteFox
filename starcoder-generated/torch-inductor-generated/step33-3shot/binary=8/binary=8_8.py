
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
 
    def forward(self, x1, bias):
        v1 = self.conv(x1)
        v2 = v1 + bias
        return v2

# Initializing the model
m = Model()
nn.init.zeros_(m.conv.weight)
nn.init.zeros_(m.conv.bias)

# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
bias = torch.randn(1, 8, 64, 64)
