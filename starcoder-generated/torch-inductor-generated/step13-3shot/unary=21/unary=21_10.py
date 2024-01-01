
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(4, 8, 4)
    def forward(self, x10):
        x11 = self.conv(x10)
        x12 = torch.tanh(x11)
        return x12
# Inputs to the model
x10 = torch.randn(256, 4, 224, 224)
