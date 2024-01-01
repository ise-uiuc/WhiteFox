
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 10, (2, 1), padding=(1, 2), dilation=(3, 2))
    def forward(self, x):
        v = self.conv(x)
        v = x
        v = torch.tanh(v)
        return v
# Inputs to the model
x = torch.randn(1, 3, 64, 24)
