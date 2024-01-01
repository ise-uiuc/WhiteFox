
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self._conv1 = torch.nn.Conv2d(in_channels=1, out_channels=2, kernel_size=(2, 2))
    def forward(self, x1):
        y1 = self._conv1(x1)
        t1 = torch.tanh(y1)
        return t1
# Inputs to the model
x1 = torch.randn(4, 1, 4, 8)
