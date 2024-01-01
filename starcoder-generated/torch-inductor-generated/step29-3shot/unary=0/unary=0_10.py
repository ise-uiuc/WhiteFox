
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 3, stride=3, padding=1)
    def forward(self, x4):
        v1 = self.conv(x4)
        y1 = v1 * 0.5
        y2 = v1 * v1
        y3 = y2 * v1
        y4 = y3 * 0.044715
        v2 = v1 + y4
        y5 = v2 * 0.7978845608028654
        y6 = torch.tanh(y5)
        y7 = y6 + 1
        v3 = y3 * y7
        return v3
# Inputs to the model
x4 = torch.randn(1, 3, 25, 25)
