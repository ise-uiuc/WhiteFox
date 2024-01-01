
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 64, 3, stride=2, padding=(1, 1), bias=False)
    def forward(self, x):
        o1 = self.conv(x)
        o2 = torch.tanh(o1)
        return o2
# Inputs to the model
x = torch.randn(1, 3, 64, 64)
