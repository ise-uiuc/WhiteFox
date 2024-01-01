
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super(ModelTanh, self).__init__()
        self.conv = torch.nn.Conv2d(3, 8, 1, stride=2, padding=1)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = torch.tanh(3*v1)
        return v2
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
