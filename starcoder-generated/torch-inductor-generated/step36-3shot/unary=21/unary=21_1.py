
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 1, 1, stride=1, groups=1)
    def forward(self, x):
        a0 = self.conv(x)
        a1 = torch.tanh(a0)
        return a1
# Inputs to the model
x = torch.randn(1, 1, 8, 8)
