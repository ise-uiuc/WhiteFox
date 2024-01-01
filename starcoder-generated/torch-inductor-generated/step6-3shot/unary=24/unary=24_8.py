
class Model(torch.nn.Module):
    def __init__(self, negative_slope):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
        self.negative_slope = negative_slope
        self.activation = torch.nn.ReLU(self.negative_slope)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = self.activation(v1)
        return v2
negative_slope = 0.1
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
