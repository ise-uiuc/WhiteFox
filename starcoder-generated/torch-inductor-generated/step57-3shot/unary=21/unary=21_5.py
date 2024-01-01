
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(16, 16, 1, stride=1, padding=0, dilation=2, groups=1)
    def forward(self, x):
        v1 = self.conv(x)
        t1 = torch.tanh(v1)
        return t1
# Inputs to the model
x = torch.randn(1, 16, 10, 10)
