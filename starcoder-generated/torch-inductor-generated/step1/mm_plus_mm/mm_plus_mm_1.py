
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 3, stride=1, padding=1)

    def forward(self, x, y):
        v1 = self.conv(x)
        v2 = v1.mm(y)
        v3 = y.mm(x)
        v4 = v2 + v3
        return v4

# Initializing the model
m = Model()

# Inputs to the model - x will be replaced by an input tensor and y will be replaced by another input tensor
x = torch.randn(1, 3, 64, 64)
y = torch.randn(1, 8, 64, 64)
