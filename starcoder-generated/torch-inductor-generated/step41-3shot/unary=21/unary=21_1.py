
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(in_channels=3, out_channels=43, kernel_size=(1, 1))
    def forward(self, x1):
        y1 = self.conv(x1)
        t1 = torch.tanh(y1)
        return t1
# Inputs to the model
x1 = torch.randn(2, 3, 6, 4)
