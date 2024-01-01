
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(16, 32, 3, stride=2, padding=1)
    def forward(self, x1111):
        x1112 = self.conv(x1111)
        x1113 = torch.tanh(x1112)
        return x1113
# Inputs to the model
x1111 = torch.randn(1, 16, 256, 256)
