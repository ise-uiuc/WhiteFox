
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super(ModelTanh, self).__init__()
        self.conv = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
    def forward(self, x1):
        y1 = self.conv(x1)
        t1 = torch.tanh(y1)
        return t1
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
