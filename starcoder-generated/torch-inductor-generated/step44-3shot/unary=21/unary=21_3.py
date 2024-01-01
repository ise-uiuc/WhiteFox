
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 3, stride=2, padding=1, dilation=1)
    def forward(self, x):
        y1 = self.conv(x)
        y2 = torch.tanh(y1)
        y3 = torch.tanh(y2)
        y4 = torch.tanh(y3)
        return y4
# Inputs to the model
x = torch.randn(1, 3, 32, 32)
