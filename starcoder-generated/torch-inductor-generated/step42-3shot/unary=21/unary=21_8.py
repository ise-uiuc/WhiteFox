
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super(ModelTanh, self).__init__()
        self.conv = torch.nn.Conv2d(2, 2, 1, stride=1)
    def forward(self, x):
        v1 = self.conv(x)
        v2 = torch.tanh(v1)
        return v2
# Inputs to the model
x = torch.randn(1, 2, 8, 8)
