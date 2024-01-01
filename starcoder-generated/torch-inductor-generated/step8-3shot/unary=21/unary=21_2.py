
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super(ModelTanh, self).__init__()
        self.conv = torch.nn.Conv2d(128, 64, 3, stride=1, padding=1)
    def forward(self, x1):
        r1 = self.conv(x1)
        r2 = torch.tanh(r1)
        return r2
# Inputs to the model
x1 = torch.randn(10, 128, 16, 16)
