
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 1, 1, dilation=1, padding=0, stride=1)
    def forward(self, x):
        n1 = self.conv(x)
        n2 = torch.tanh(n1)
        return n2
# Inputs to the model
x = torch.randn(1, 3, 25, 25)
