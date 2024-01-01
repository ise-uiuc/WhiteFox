
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 3, 2, stride=2)
    def forward(self, x1):
        t1 = torch.tanh(x1)
        y1 = self.conv(t1)
        return y1
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
