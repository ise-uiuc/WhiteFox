
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(32, 96, 3, stride=2)
        self.conv2 = torch.nn.Conv2d(96, 168, 1, stride=2)
    def forward(self, x57):
        v8 = self.conv(x57)
        v9 = torch.tanh(v8)
        return self.conv2(v9)
# Inputs to the model
x57 = torch.randn(16, 32, 15, 169)
