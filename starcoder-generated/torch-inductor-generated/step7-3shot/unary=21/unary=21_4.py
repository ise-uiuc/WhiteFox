
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 32, 3, stride=5, padding=7, dilation=5)
    def forward(self, x):
        v1 = self.conv(x)
        tanh = torch.tanh(v1)
        return tanh
# Inputs to the model
x = torch.randn(2, 3, 224, 224)
