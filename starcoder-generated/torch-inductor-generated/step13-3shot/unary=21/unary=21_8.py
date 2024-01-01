
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(10, 1, 14, stride=1)
    def forward(self, x):
        x1 = self.conv(x)
        x2 = torch.tanh(x1)
        return x2
# Inputs to the model
x = torch.randn(30, 10, 15, 16)
