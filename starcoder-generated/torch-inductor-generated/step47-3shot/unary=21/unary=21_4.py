
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 64, 1)
        self.conv2 = torch.nn.Conv2d(64, 64, 1)

    def forward(self, t0):
        y0 = self.conv(t0)
        y1 = torch.tanh(y0)
        y2 = self.conv2(y1)
        return y2
# Inputs to the model
t0 = torch.randn(1, 1, 64, 64)
