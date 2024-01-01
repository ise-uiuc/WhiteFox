
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
        self.bias = torch.nn.Parameter(torch.zeros(1, 8, 1, 1))
    def forward(self, input_x):
        v1 = self.conv(input_x)
        v2 = v1.add(self.bias)
        v3 = v2.clamp(min=0, max=6)
        v4 = v3.div(6)
        return v4
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
