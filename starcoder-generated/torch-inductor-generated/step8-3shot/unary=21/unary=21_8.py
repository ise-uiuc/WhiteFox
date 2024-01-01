
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 16, 3, stride=1)
        self.sigmoid = torch.nn.Sigmoid()
    def forward(self, x):
        y = self.conv(x)
        x1 = self.sigmoid(y)
        v2 = torch.tanh(x1)
        return v2
# Inputs to the model
x1 = torch.randn(64, 3, 64, 64)
